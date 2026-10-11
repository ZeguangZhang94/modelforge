#!/usr/bin/env python3
"""safetensors → GGUF 转换管线。

现实约束：转换需要 llama.cpp 的 convert_hf_to_gguf.py + torch（数 GB 依赖）。
本模块做三件事：
1. 检测本机是否具备转换条件（脚本 + torch）
2. 具备 → 直接执行转换（输出 F16 GGUF，可选再量化）
3. 不具备 → 给出精确的引导（去哪下现成 GGUF / 怎么装环境）

用法：python -m modelforge convert <模型目录> [--out 输出目录] [--quant q4_k_m]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

GUIDE = """\
转换条件不具备。你有两个选择：

【方案 A：直接下载现成 GGUF（推荐，省事）】
  绝大多数热门模型都有社区量化好的 GGUF：
  - 打包台「模型商店」里直接选（全部已验证可下载）
  - 或去 HuggingFace/ModelScope 搜 "<模型名> GGUF"（认准 bartowski / unsloth / 官方）

【方案 B：自己转换（需要一次性配置环境）】
  1. 安装 Python 依赖：pip install torch transformers gguf
  2. 获取转换脚本：git clone https://github.com/ggml-org/llama.cpp
  3. 设置环境变量后重试：
     set LLAMA_CPP_HOME=C:\\path\\to\\llama.cpp   (Windows)
     export LLAMA_CPP_HOME=/path/to/llama.cpp     (Linux)
  4. 重新运行：python -m modelforge convert <模型目录>
"""


def find_convert_script() -> Path | None:
    home = os.environ.get("LLAMA_CPP_HOME", "")
    if home:
        cand = Path(home) / "convert_hf_to_gguf.py"
        if cand.exists():
            return cand
        cand2 = Path(home) / "convert-hf-to-gguf.py"
        if cand2.exists():
            return cand2
    return None


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def convert(model_dir: str | Path, out_dir: str | Path | None = None,
            quant: str = "") -> int:
    model_dir = Path(model_dir)
    out_dir = Path(out_dir) if out_dir else model_dir
    if not (model_dir / "config.json").exists():
        print(f"[错误] {model_dir} 不是 HuggingFace 模型目录（缺 config.json）")
        return 1

    script = find_convert_script()
    if not script or not torch_available():
        if not script:
            print("[检测] 未找到 llama.cpp 转换脚本（LLAMA_CPP_HOME 未设置）")
        if not torch_available():
            print("[检测] 未安装 torch")
        print()
        print(GUIDE)
        return 2

    out_gguf = out_dir / f"{model_dir.name}-f16.gguf"
    print(f"转换 {model_dir} → {out_gguf.name}（F16，需要几分钟到几十分钟）...")
    r = subprocess.run(
        [sys.executable, str(script), str(model_dir), "--outfile", str(out_gguf)],
        timeout=7200)
    if r.returncode != 0 or not out_gguf.exists():
        print("[失败] 转换脚本执行失败，详见上方输出")
        return 1
    print(f"✔ 转换完成: {out_gguf}")

    if quant:
        quantize_bin = os.environ.get("LLAMA_CPP_HOME", "")
        exe = Path(quantize_bin) / ("llama-quantize.exe" if os.name == "nt"
                                    else "llama-quantize")
        if exe.exists():
            q_out = out_dir / f"{model_dir.name}-{quant.upper()}.gguf"
            print(f"量化到 {quant.upper()} ...")
            r = subprocess.run([str(exe), str(out_gguf), str(q_out), quant.upper()],
                               timeout=7200)
            if r.returncode == 0:
                print(f"✔ 量化完成: {q_out}")
                out_gguf.unlink()  # 删掉中间产物
                return 0
            print("[警告] 量化失败，保留 F16 版本")
        else:
            print(f"[提示] 未找到 llama-quantize，保留 F16 版本: {out_gguf}")
    return 0
