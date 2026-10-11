#!/usr/bin/env python3
"""资源规划器：三级分布计算（VRAM 热 / RAM 温 / SSD 冷）。

吸收并泛化 Strata 的设计（docs/02-架构设计.md 第 3.4 节）：
- MoE：共享层 + 热专家缓存进 VRAM，全量专家池驻 RAM（pinned），溢出走 SSD 流式
- 稠密：层卸载（n_gpu_layers），KV cache 量化按剩余显存选择
- 多卡：同型号卡聚合显存池，生成 tensor-split 比例

纯函数、无副作用，可直接单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field

VRAM_UTILIZATION = 0.85   # 显存预算利用率上限
SYSTEM_RAM_RESERVE_GB = 6.0


@dataclass
class DensePlan:
    n_gpu_layers: int
    total_layers: int
    kv_quant: str                 # f16 | q8_0 | q4_0
    vram_used_gb: float
    ram_used_gb: float
    offload_ratio: float          # 0~1，1 = 全在显存
    speed_class: str              # fast | medium | slow
    warnings: list[str] = field(default_factory=list)


@dataclass
class MoePlan:
    expert_cache_gb: float        # VRAM 中给热专家缓存的容量
    ram_pool_gb: float            # RAM 中全量专家池
    ssd_stream_gb: float          # 需从 SSD 流式读的部分
    kv_quant: str
    speed_class: str
    warnings: list[str] = field(default_factory=list)


def _kv_quant_for(vram_left_gb: float) -> str:
    if vram_left_gb >= 2.0:
        return "f16"
    if vram_left_gb >= 1.0:
        return "q8_0"
    return "q4_0"


def plan_dense(model_size_gb: float, total_layers: int,
               vram_free_gb: float, ram_free_gb: float,
               context: int) -> DensePlan:
    """稠密模型：预算内尽量多层上卡。"""
    warnings: list[str] = []
    if total_layers <= 0:
        total_layers = 1
    kv_gb = max(0.5, context / 8192 * 0.5) * (model_size_gb / 4.9)  # 随规模缩放
    budget = vram_free_gb * VRAM_UTILIZATION
    per_layer = model_size_gb * 0.9 / total_layers
    shared = model_size_gb * 0.1  # embedding/输出等共享张量

    if budget >= model_size_gb + kv_gb:
        n = total_layers
    elif budget > shared + kv_gb:
        n = min(total_layers, int((budget - shared - kv_gb) / per_layer))
    else:
        n = 0
        warnings.append("显存预算不足以放下共享层，全部走 CPU（很慢）")

    vram_left = max(0.0, budget - shared - n * per_layer)
    kv_quant = _kv_quant_for(vram_left) if n > 0 else "q4_0"
    if kv_quant != "f16":
        warnings.append(f"显存紧张，KV cache 降为 {kv_quant}（轻微质量损失换更长上下文）")

    offload = n / total_layers
    ram_used = model_size_gb * (1 - offload) + 2.0
    if ram_used + SYSTEM_RAM_RESERVE_GB > ram_free_gb:
        warnings.append("内存余量不足，运行时请关闭浏览器等大内存程序")

    speed = "fast" if offload >= 0.95 else "medium" if offload >= 0.5 else "slow"
    return DensePlan(
        n_gpu_layers=n, total_layers=total_layers, kv_quant=kv_quant,
        vram_used_gb=round(shared + n * per_layer + kv_gb, 2),
        ram_used_gb=round(ram_used, 2),
        offload_ratio=round(offload, 2), speed_class=speed, warnings=warnings,
    )


def plan_moe(total_size_gb: float, active_ratio: float,
             vram_free_gb: float, ram_free_gb: float) -> MoePlan:
    """MoE 模型（Strata 式三级分布）。

    active_ratio = 单 token 激活参数 / 总参数（如 10/24576 个专家）
    """
    warnings: list[str] = []
    budget = vram_free_gb * VRAM_UTILIZATION
    # 热专家缓存：预算的 70% 给专家缓存，30% 给共享层 + KV
    expert_cache_gb = round(max(0.0, budget * 0.70), 2)

    ram_available = max(0.0, ram_free_gb - SYSTEM_RAM_RESERVE_GB)
    if ram_available >= total_size_gb:
        ram_pool, ssd = total_size_gb, 0.0
    else:
        ram_pool = ram_available
        ssd = round(total_size_gb - ram_available, 2)
        warnings.append(
            f"内存放不下全量专家池，{ssd:.0f} GB 将从 SSD 流式读取"
            "（速度显著下降，强烈建议升级内存或选更小档位）")

    kv_quant = _kv_quant_for(budget * 0.30)
    # 命中率估算：对比"热工作集"而非全量池——MoE 单 token 只激活少量专家，
    # 且热度高度集中（参考 Strata：缓存覆盖率 ~50% 即可达全速的 90%+）
    hot_working_set = total_size_gb * min(0.25, active_ratio * 5)
    hit_rate_est = min(0.95, expert_cache_gb / max(hot_working_set, 0.1))
    if ssd > 0:
        speed = "slow"
    elif hit_rate_est >= 0.6:
        speed = "fast"
    else:
        speed = "medium"
        warnings.append("热专家缓存覆盖率偏低，建议更大显存或更小档位")

    return MoePlan(
        expert_cache_gb=expert_cache_gb,
        ram_pool_gb=round(ram_pool, 2), ssd_stream_gb=ssd,
        kv_quant=kv_quant, speed_class=speed, warnings=warnings,
    )


def tensor_split_ratio(gpus: list[dict]) -> str:
    """多卡切分比例（llama.cpp --tensor-split 格式），按可用显存比例分配。

    仅同型号卡组调用（安装器已保证）。返回如 "1.0,1.0" 或 "2.0,1.0"。
    """
    if not gpus:
        return ""
    frees = [max(g["vram_total_gb"] - g["vram_used_gb"], 0.1) for g in gpus]
    base = min(frees)
    return ",".join(f"{f / base:.1f}" for f in frees)


def to_engine_args(plan: DensePlan | MoePlan, context: int) -> list[str]:
    """把规划结果翻译成 llama.cpp 系引擎参数。"""
    args = ["--ctx-size", str(context)]
    if isinstance(plan, DensePlan):
        args += ["--n-gpu-layers", str(plan.n_gpu_layers)]
    if plan.kv_quant != "f16":
        args += ["--cache-type-k", plan.kv_quant, "--cache-type-v", plan.kv_quant]
    return args
