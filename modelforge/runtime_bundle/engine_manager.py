#!/usr/bin/env python3
"""引擎管理器：推理引擎（llama-server 等）的获取、参数计算、进程生命周期。

职责（对应 docs/02-架构设计.md 第 3.4/3.5 节）：
- 解析引擎二进制：安装包内置 → 本地缓存 → 按 version_pin 下载（可选）
- 按可用显存自动计算 n_gpu_layers（稠密层卸载调优）
- 拉起引擎子进程、等待健康、崩溃自动重启（带退避与上限）
- 优雅停止

纯标准库。Windows / Linux / macOS 通用。
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

INTERNAL_PORT = 18091  # 默认引擎内部端口（被占用时自动换端口）


def find_free_port(preferred: int = INTERNAL_PORT) -> int:
    """从偏好端口起找空闲端口，避免残留进程占用导致健康检查误判。"""
    import socket
    for p in range(preferred, preferred + 50):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    raise EngineError("无可用内部端口")

# llama.cpp 官方 release 资产名模板：按 (平台, 显卡厂商) 选择
# 已对照 b11526 实际资产列表校准（2026-10）：
#   Windows AMD 优先 Vulkan（33MB，免 ROCm 环境）；ROCm 10.0 Windows 版也存在（258MB，备选）
#   Windows NVIDIA 自 b4xxx 起 CUDA 版需同时下载 cudart 伴随包
#   Linux/macOS 资产为 tar.gz，Windows 为 zip
_LLAMA_ASSET = {
    ("Windows", "nvidia"): "llama-{tag}-bin-win-cuda-12.4-x64.zip",
    ("Windows", "amd"):    "llama-{tag}-bin-win-vulkan-x64.zip",
    ("Windows", "intel"):  "llama-{tag}-bin-win-sycl-x64.zip",
    ("Windows", "cpu"):    "llama-{tag}-bin-win-cpu-x64.zip",
    ("Linux", "nvidia"):   "llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz",
    ("Linux", "amd"):      "llama-{tag}-bin-ubuntu-rocm-10.0-x64.tar.gz",
    ("Linux", "intel"):    "llama-{tag}-bin-ubuntu-sycl-fp16-x64.tar.gz",
    ("Linux", "cpu"):      "llama-{tag}-bin-ubuntu-x64.tar.gz",
    ("Darwin", "cpu"):     "llama-{tag}-bin-macos-arm64.tar.gz",  # Apple Silicon 内置 Metal
}

# NVIDIA CUDA 版必需的运行时伴随包（与主包一同解压）
_LLAMA_CUDART = {
    ("Windows", "nvidia"): "cudart-llama-{tag}-bin-win-cuda-12.4-x64.zip",
    ("Linux", "nvidia"): "cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz",
}


class EngineError(RuntimeError):
    pass


# ---------------- GGUF 层数读取（独立小实现，避免跨包依赖） ----------------

def gguf_block_count(model_path: Path) -> int:
    """从 GGUF 元数据读取 block_count（层数），失败返回 0。"""
    try:
        with open(model_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return 0
            (version,) = struct.unpack("<I", f.read(4))
            if version not in (2, 3):
                return 0
            f.read(8)
            (kv_count,) = struct.unpack("<Q", f.read(8))

            def rstr() -> str:
                (n,) = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            arch = "llama"
            for _ in range(kv_count):
                key = rstr()
                (vt,) = struct.unpack("<I", f.read(4))
                if key == "general.architecture" and vt == 8:
                    arch = rstr()
                    continue
                if key.endswith(".block_count") and vt == 4:
                    (n,) = struct.unpack("<I", f.read(4))
                    return n
                # 跳过值（0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool,10 u64,11 i64,12 f64）
                fmt = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                       10: 8, 11: 8, 12: 8}
                if vt == 8:
                    rstr()
                elif vt == 9:
                    (et,) = struct.unpack("<I", f.read(4))
                    (cnt,) = struct.unpack("<Q", f.read(8))
                    if et == 8:
                        for _ in range(cnt):
                            rstr()
                    elif et in fmt:
                        f.read(fmt[et] * cnt)
                    else:
                        return 0
                elif vt in fmt:
                    f.read(fmt[vt])
                else:
                    return 0
    except Exception:
        return 0
    return 0


# ---------------- 引擎获取 ----------------

def find_engine_binary(search_dirs: list[Path]) -> Path | None:
    names = (["llama-server.exe"] if platform.system() == "Windows" else []) + \
            ["llama-server", "llama-server.py"]
    for d in search_dirs:
        if not d.exists():
            continue
        for exe in names:
            cand = d / exe
            if cand.exists():
                return cand
            for p in d.rglob(exe):
                return p
    return None


def download_engine(tag: str, vendor: str, dest_dir: Path,
                    mirror: str = "https://github.com") -> Path:
    """从 llama.cpp release 下载引擎二进制（带断点续传），返回二进制路径。

    vendor: nvidia | amd | intel | cpu —— 决定下载 CUDA / Vulkan / CPU 版。
    """
    asset_tpl = _LLAMA_ASSET.get((platform.system(), vendor)) or \
        _LLAMA_ASSET.get((platform.system(), "cpu"))
    if not asset_tpl:
        raise EngineError(f"平台 {platform.system()}/{vendor} 暂无预置引擎资产模板")
    asset = asset_tpl.format(tag=tag)
    base = f"{mirror}/ggml-org/llama.cpp/releases/download/{tag}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloads = [asset]
    cudart_tpl = _LLAMA_CUDART.get((platform.system(), vendor))
    if cudart_tpl:
        downloads.append(cudart_tpl.format(tag=tag))

    for name in downloads:
        url = f"{base}/{name}"
        arc = dest_dir / name
        part = arc.with_suffix(arc.suffix + ".part")
        headers, mode = {}, "wb"
        if part.exists():
            headers["Range"] = f"bytes={part.stat().st_size}-"
            mode = "ab"
        print(f"  下载引擎 {name} ...")
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=60)
        # 断点续传校验：服务端必须真的接受 Range（206），否则从头重来
        if mode == "ab" and resp.status != 206:
            print(f"  [提示] 服务端未接受断点续传，重新下载 {name}")
            resp.close()
            resp = urllib.request.urlopen(
                urllib.request.Request(url), timeout=60)
            mode = "wb"
        with resp, open(part, mode) as f:
            shutil.copyfileobj(resp, f, 1 << 20)
        part.rename(arc)
        _extract_archive(arc, dest_dir)

    exe = find_engine_binary([dest_dir])
    if not exe:
        raise EngineError("解压后未找到 llama-server 可执行文件")
    if platform.system() != "Windows":
        exe.chmod(0o755)
    return exe


def _extract_archive(arc: Path, dest_dir: Path) -> None:
    """按扩展名解 zip 或 tar.gz。"""
    if arc.suffix == ".zip":
        with zipfile.ZipFile(arc) as zf:
            zf.extractall(dest_dir)
    elif arc.name.endswith(".tar.gz") or arc.suffix == ".tgz":
        import tarfile
        with tarfile.open(arc) as tf:
            tf.extractall(dest_dir)
    else:
        raise EngineError(f"未知的引擎压缩包格式: {arc.name}")


# ---------------- 层卸载计算 ----------------

def compute_gpu_layers(model_path: Path, model_size_gb: float,
                       gpu_free_vram_gb: float, context: int,
                       utilization: float = 0.85) -> int:
    """稠密模型层卸载自动调优：预算内能放多少层放多少层。

    预算 = 可用显存 × 利用率 - KV cache 预留
    每层显存 ≈ (模型大小 × 0.9) / 层数   （0.9 扣除共享张量）
    """
    layers = gguf_block_count(model_path)
    if layers <= 0:
        return 999 if gpu_free_vram_gb >= model_size_gb else 0  # 读不出层数时按整体判断
    kv_reserve = max(0.5, context / 8192 * 0.5)  # 粗估 KV 预留
    budget = gpu_free_vram_gb * utilization - kv_reserve
    if budget <= 0:
        return 0
    per_layer = model_size_gb * 0.9 / layers
    n = int(budget / per_layer)
    return max(0, min(layers, n))


# ---------------- 进程生命周期 ----------------

class EngineManager:
    """引擎进程看门狗：拉起、健康等待、崩溃重启、优雅停止。"""

    def __init__(self, binary: Path, args: list[str], log_path: Path,
                 max_restarts: int = 5, port: int = INTERNAL_PORT):
        self.binary = binary
        self.args = args
        self.log_path = log_path
        self.max_restarts = max_restarts
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.restarts = 0

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"

    def is_healthy(self, timeout: float = 2.0) -> bool:
        try:
            with urllib.request.urlopen(self.health_url(), timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def _cmd(self) -> list[str]:
        # 支持 Python 脚本形态的引擎（测试桩 / Python 封装引擎）
        if self.binary.suffix == ".py":
            return [sys.executable, str(self.binary), *self.args]
        return [str(self.binary), *self.args]

    def start(self, wait_s: float = 120.0) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            self._cmd(),
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise EngineError(
                    f"引擎进程退出（code={self.proc.returncode}），详见 {self.log_path}")
            if self.is_healthy():
                return
            time.sleep(1.0)
        raise EngineError(f"引擎启动超时（{wait_s:.0f}s），详见 {self.log_path}")

    def ensure_running(self) -> bool:
        """供监控循环调用：死了就重启，超过上限返回 False。"""
        if self.proc and self.proc.poll() is None:
            return True
        if self.restarts >= self.max_restarts:
            return False
        self.restarts += 1
        time.sleep(min(2 ** self.restarts, 30))
        try:
            self.start()
            return True
        except EngineError:
            return False

    def stop(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()


def build_engine_args(config: dict, gpu_free_vram_gb: float,
                      port: int = INTERNAL_PORT) -> list[str]:
    """由安装配置生成 llama-server 启动参数。

    统一走 planner 资源规划（docs/02-架构设计.md 第 3.4 节）：
    - 稠密：层卸载 n_gpu_layers + KV cache 量化
    - MoE：--n-cpu-moe 专家卸载（Strata 式 VRAM/RAM 分布的 llama.cpp 实现）
    - 投机解码：有草稿模型用 --spec-draft-model；否则 ngram-simple（零成本提速）
    - 多卡：--tensor-split
    """
    from planner import plan_dense, plan_moe, to_engine_args

    model = Path(config["model_file"])
    defaults = config.get("engine", {}).get("defaults", {})
    context = int(defaults.get("context", 8192))
    size_gb = model.stat().st_size / 1e9 if model.exists() else 1.0
    layers = int(config.get("layers", 0)) or gguf_block_count(model)
    ram_free = float(config.get("ram_free_gb", 32.0))

    args = [
        "--model", str(model),
        "--port", str(port),
        "--host", "127.0.0.1",
        "--parallel", str(int(defaults.get("parallel", 1))),
        "--metrics",
    ]

    if config.get("architecture") == "moe" and int(config.get("expert_count", 0)) > 0:
        active = config["expert_used_count"] / config["expert_count"]
        plan = plan_moe(size_gb, active, gpu_free_vram_gb, ram_free)
        args += to_engine_args(plan, context)
        # 专家卸载：缓存覆盖率 → 留在 CPU 的专家层数
        coverage = min(1.0, plan.expert_cache_gb /
                       max(size_gb * min(0.25, active * 5), 0.1))
        if plan.ssd_stream_gb > 0 or coverage < 0.3:
            args += ["--cpu-moe"]          # 显存极小：专家全留 CPU
        elif coverage < 0.9 and layers > 0:
            n_cpu = max(1, int(layers * (1 - coverage)))
            args += ["--n-cpu-moe", str(n_cpu)]  # 部分专家层留 CPU
        # MoE 共享层与注意力仍尽量上卡
        args += ["--n-gpu-layers", "999"]
    else:
        plan = plan_dense(size_gb, layers, gpu_free_vram_gb, ram_free, context)
        args += to_engine_args(plan, context)

    # 投机解码（b11526 实测参数名：--spec-draft-model / --spec-type）
    if defaults.get("speculative_decode"):
        draft = defaults.get("draft_model", "")
        if draft and (model.parent / draft).exists():
            # 独立草稿文件（eagle3/mtp 小模型随包）
            args += ["--spec-draft-model", str(model.parent / draft),
                     "--spec-type", defaults.get("spec_type", "draft-simple")]
        else:
            # 内嵌 MTP 层或 ngram 兜底
            args += ["--spec-type", defaults.get("spec_type", "ngram-simple")]

    if config.get("tensor_split"):
        args += ["--tensor-split", config["tensor_split"]]
    mmproj = model.parent / f"mmproj-{model.name.split('-')[-1]}"
    if mmproj.exists():
        args += ["--mmproj", str(mmproj)]
    return args
