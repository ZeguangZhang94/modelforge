"""硬件需求估算器：按模型 Profile 为每个量化变体生成资源需求表。

估算公式（对应 docs/02-架构设计.md 第 2.4 节）：
- vram_hot：共享层 + KV cache + 热数据目标容量（≈ 可用显存的 85%，由安装器最终计算）
- ram_resident：MoE 为全量专家池；稠密为未卸载层 + KV 预留
- disk：权重文件 × 1.3（含临时空间）

注意：这里的数值是"打包侧预估"，安装器会在真实硬件上复核并降档。
"""

from __future__ import annotations

from dataclasses import dataclass

from .analyzer import ModelProfile
from .modelpack import Requirements

# KV cache 每 token 每层的近似字节数（fp16，按 hidden 4096 估算基准，随 embd 线性缩放）
_KV_BYTES_PER_TOKEN_LAYER_BASE = 4096 * 2 * 2  # 4096 hidden × k/v × fp16
_VRAM_UTILIZATION = 0.85


@dataclass
class Estimate:
    variant_id: str
    size_gb: float
    requirements: Requirements
    notes: str = ""


def _kv_cache_gb(profile: ModelProfile, context: int) -> float:
    embd = profile.source_config.get("embd") or 4096
    layers = profile.source_config.get("layers") or 32
    per_tok = _KV_BYTES_PER_TOKEN_LAYER_BASE * (embd / 4096)
    return per_tok * layers * context / 1e9


def estimate_variant(profile: ModelProfile, variant: dict, context: int = 8192) -> Estimate:
    """为单个量化变体估算资源需求。"""
    size_gb = float(variant["size_gb"])
    bits = float(variant.get("bits", 4.9))
    kv_gb = _kv_cache_gb(profile, context)

    if profile.architecture == "moe":
        # MoE：热专家缓存放 VRAM（目标 8~12GB 区间取可行为准），全量专家池驻 RAM
        vram_hot = min(12.0, size_gb * 0.25) + kv_gb
        ram_resident = size_gb + kv_gb  # 全量权重需进 RAM
        notes = "MoE：热专家上显卡，全量专家池驻内存"
    else:
        # 稠密：尽量全量上卡；放不下则部分卸载，其余驻 RAM
        vram_hot = size_gb + kv_gb
        ram_resident = max(4.0, (size_gb + kv_gb) * 0.5)  # 部分卸载 + 系统余量
        notes = "稠密：优先全量上卡，不足自动层卸载"

    min_ram = max(8.0, ram_resident + 6.0)      # 系统与其他程序余量 6GB
    disk = size_gb * 1.3

    return Estimate(
        variant_id=variant["id"],
        size_gb=size_gb,
        requirements=Requirements(
            min_ram_gb=round(min_ram, 1),
            recommended_ram_gb=round(min_ram * 1.5, 1),
            vram_hot_gb=round(vram_hot, 1),
            disk_gb=round(disk, 1),
        ),
        notes=notes,
    )


def estimate_all(profile: ModelProfile, context: int = 8192) -> list[Estimate]:
    return [estimate_variant(profile, v, context) for v in profile.quant_variants]


def print_estimate_table(profile: ModelProfile, estimates: list[Estimate]) -> str:
    lines = [
        f"模型: {profile.family}  架构: {profile.architecture}  "
        f"参数: {profile.params_b}B(激活 {profile.active_params_b}B)  "
        f"上下文上限: {profile.context_max}",
        "",
        f"{'变体':<12}{'大小GB':>8}{'VRAM热数据GB':>14}{'最低RAM GB':>12}{'磁盘GB':>9}  说明",
        "-" * 78,
    ]
    for e in estimates:
        r = e.requirements
        lines.append(
            f"{e.variant_id:<12}{e.size_gb:>8.1f}{r.vram_hot_gb:>14.1f}"
            f"{r.min_ram_gb:>12.1f}{r.disk_gb:>9.1f}  {e.notes}"
        )
    return "\n".join(lines)
