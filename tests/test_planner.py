#!/usr/bin/env python3
"""资源规划器单元测试（用虚拟硬件画像，无需 GPU）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "modelforge" / "runtime_bundle"))
from planner import plan_dense, plan_moe, tensor_split_ratio, to_engine_args

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  ✔ " if cond else "  ✘ ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


print("稠密模型规划：")
# RTX 4090 24GB，7B Q4 模型（~4.4GB）→ 应全量上卡
p = plan_dense(model_size_gb=4.4, total_layers=32, vram_free_gb=22, ram_free_gb=60, context=8192)
check("24GB 卡跑 7B 全量上卡", p.n_gpu_layers == 32 and p.speed_class == "fast")
check("KV 保持 f16", p.kv_quant == "f16")

# RTX 3060 12GB，34B Q4 模型（~20GB）→ 部分卸载
p = plan_dense(model_size_gb=20.0, total_layers=48, vram_free_gb=11, ram_free_gb=60, context=8192)
check("12GB 卡跑 34B 部分卸载", 0 < p.n_gpu_layers < 48, f"ngl={p.n_gpu_layers}")
check("卸载比例在 0.2~0.6", 0.2 <= p.offload_ratio <= 0.6, f"ratio={p.offload_ratio}")
check("速度档 slow（卸载比 < 0.5）", p.speed_class == "slow")

# 无独显 → 纯 CPU
p = plan_dense(model_size_gb=4.4, total_layers=32, vram_free_gb=0, ram_free_gb=60, context=8192)
check("无显卡走 CPU（slow）", p.n_gpu_layers == 0 and p.speed_class == "slow")

args = to_engine_args(p, 8192)
check("引擎参数含 n-gpu-layers 0", "--n-gpu-layers" in args and "0" in args)

print("MoE 模型规划（Strata 式三级分布）：")
# 125B MoE，激活比 ~4%，24GB 卡 + 66GB 空闲内存 → 经典 Strata 场景（64GB 内存机器）
p = plan_moe(total_size_gb=58.0, active_ratio=0.04, vram_free_gb=22, ram_free_gb=66)
check("热专家缓存占预算 70%", abs(p.expert_cache_gb - 22 * 0.85 * 0.70) < 0.2,
      f"{p.expert_cache_gb} GB")
check("全量专家池驻 RAM、无 SSD 流式", p.ssd_stream_gb == 0 and p.speed_class == "fast")

# 同模型但 32GB 内存 → SSD 兜底 + 警告
p = plan_moe(total_size_gb=58.0, active_ratio=0.04, vram_free_gb=22, ram_free_gb=30)
check("内存不足触发 SSD 流式", p.ssd_stream_gb > 20 and p.speed_class == "slow")
check("给出升级内存警告", any("内存" in w for w in p.warnings))

print("多卡切分：")
gpus = [
    {"name": "RTX 3090", "vram_total_gb": 24, "vram_used_gb": 1},
    {"name": "RTX 3090", "vram_total_gb": 24, "vram_used_gb": 1},
]
check("同型号双卡等比切分", tensor_split_ratio(gpus) == "1.0,1.0")
gpus[1]["vram_used_gb"] = 13  # 第二张卡被占了一半
check("占用不均按空闲显存比例分配", tensor_split_ratio(gpus) == "2.1,1.0",
      tensor_split_ratio(gpus))
check("无卡返回空", tensor_split_ratio([]) == "")

print()
if failures:
    print(f"失败 {len(failures)} 项: {failures}")
    sys.exit(1)
print("全部通过 ✔")
