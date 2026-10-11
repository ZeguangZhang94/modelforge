#!/usr/bin/env python3
"""M3 引擎参数生成 + Linux tar.gz 解压路径 单元测试。"""

import io
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "modelforge" / "runtime_bundle"))
from engine_manager import build_engine_args, _extract_archive

failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  ✔ " if cond else "  ✘ ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def fake_config(**kw):
    cfg = {
        "name": "t", "model_file": str(Path(__file__).parent / "fake_model" / "tiny-llama-q8_0.gguf"),
        "engine": {"defaults": {"context": 4096, "speculative_decode": False, "parallel": 1}},
        "architecture": "dense", "layers": 8, "expert_count": 0, "expert_used_count": 0,
        "gpu_free_vram_gb": 24.0, "ram_free_gb": 64.0, "tensor_split": "",
    }
    cfg.update(kw)
    return cfg


print("稠密模型引擎参数：")
args = build_engine_args(fake_config(), 24.0)
check("24GB 显存 8 层全上卡", "--n-gpu-layers" in args and
      args[args.index("--n-gpu-layers") + 1] == "8", str(args))
check("默认不开投机解码", "--spec-type" not in args)

args = build_engine_args(fake_config(), 24.0)
args_spec = build_engine_args(
    fake_config(engine={"defaults": {"context": 4096, "speculative_decode": True}}), 24.0)
check("开启投机解码 → ngram-simple", "--spec-type" in args_spec and
      args_spec[args_spec.index("--spec-type") + 1] == "ngram-simple")

args_ts = build_engine_args(fake_config(tensor_split="1.0,1.0"), 48.0)
check("多卡 tensor-split 透传", "--tensor-split" in args_ts)

print("MoE 模型引擎参数（Strata 式专家卸载）：")
moe_cfg = fake_config(architecture="moe", layers=48,
                      expert_count=128, expert_used_count=8)
moe_args = build_engine_args(moe_cfg, 24.0)
check("MoE 共享层全上卡", "--n-gpu-layers" in moe_args and
      moe_args[moe_args.index("--n-gpu-layers") + 1] == "999")
# 24GB 卡 + 充足内存 → 覆盖率高 → 不加 cpu-moe
check("显存充足时专家全上卡", "--cpu-moe" not in moe_args and "--n-cpu-moe" not in moe_args)

# 用稀疏文件模拟 20GB MoE 模型（NTFS 稀疏文件，秒建不占磁盘）
with tempfile.TemporaryDirectory() as td:
    big = Path(td) / "big-moe.gguf"
    with open(big, "wb") as f:
        f.seek(20 * 1024 ** 3 - 1)
        f.write(b"\0")
    moe_big = fake_config(architecture="moe", layers=48, expert_count=128,
                          expert_used_count=8, model_file=str(big))
    a4 = build_engine_args(moe_big, 4.0)
    check("20GB MoE + 4GB 显存 → 部分专家层卸载",
          "--n-cpu-moe" in a4 or "--cpu-moe" in a4,
          " ".join(x for x in a4 if "moe" in x or "gpu-layers" in x))
    a1 = build_engine_args(moe_big, 1.0)
    check("20GB MoE + 1GB 显存 → 专家全留 CPU", "--cpu-moe" in a1)

print("Linux/macOS tar.gz 解压路径：")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # 构造 tar.gz
    tgz = td / "llama-b11526-bin-ubuntu-x64.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        data = b"#!/bin/sh\necho mock\n"
        info = tarfile.TarInfo("llama-server")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    dest = td / "out"
    dest.mkdir()
    _extract_archive(tgz, dest)
    check("tar.gz 解压成功", (dest / "llama-server").exists())

    # zip 路径不回归
    zp = td / "test.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("bin/llama-server.exe", "mock")
    dest2 = td / "out2"
    dest2.mkdir()
    _extract_archive(zp, dest2)
    check("zip 解压不回归", (dest2 / "bin" / "llama-server.exe").exists())

print()
if failures:
    print(f"失败 {len(failures)} 项: {failures}")
    sys.exit(1)
print("全部通过 ✔")
