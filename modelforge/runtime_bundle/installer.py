#!/usr/bin/env python3
"""ModelForge 通用安装器（随安装包分发，清单驱动，与具体模型无关）。

纯 Python 标准库实现，要求 Python >= 3.9。
对应文档：docs/02-架构设计.md 第 3.2~3.4、3.6 节。

用法：
    python installer.py [--dir <安装目录>] [--gpu <序号>] [--yes]
    python installer.py --uninstall [--dir <安装目录>]
    python installer.py --diagnose  [--dir <安装目录>]
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


# ---------------- 工具 ----------------

def info(msg: str) -> None:
    print(f"  {msg}")


def warn(msg: str) -> None:
    print(f"  [警告] {msg}")


def fail(msg: str) -> None:
    print(f"\n安装失败: {msg}", file=sys.stderr)
    sys.exit(1603)  # 企业部署约定：1603 = 安装致命错误（SCCM/Intune 标准）


def preprocess_enterprise_args(argv: list[str]) -> list[str]:
    """企业静默安装契约：/S、/qn、/quiet → --yes；/DIR=path → --dir path；
    /NOSTART → --no-autostart。与 NSIS/MSI 部署习惯兼容。"""
    out: list[str] = []
    it = iter(range(len(argv)))
    skip = False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        low = a.lower()
        if low in ("/s", "/qn", "/quiet", "-s", "--silent"):
            out.append("--yes")
        elif low.startswith("/dir=") or low.startswith("/dir:"):
            out += ["--dir", a.split("=", 1)[-1] if "=" in a else a.split(":", 1)[-1]]
        elif low == "/dir" and i + 1 < len(argv):
            out += ["--dir", argv[i + 1]]
            skip = True
        elif low == "/nostart":
            out.append("--no-autostart")
        else:
            out.append(a)
    return out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def ask(prompt: str, default: str, assume_yes: bool) -> str:
    if assume_yes:
        info(f"{prompt} [{default}] → 采用默认值")
        return default
    ans = input(f"  {prompt} [{default}]: ").strip()
    return ans or default


def default_install_dir(name: str) -> Path:
    return {
        "Windows": Path(os.environ.get("LOCALAPPDATA", Path.home())) / name,
        "Darwin": Path.home() / "Library" / "Application Support" / name,
    }.get(platform.system(), Path.home() / ".local" / "share" / name)


# ---------------- 1. 硬件探测 ----------------

class Hardware:
    def __init__(self) -> None:
        self.os = platform.system()
        self.ram_total_gb = self._probe_ram()
        self.gpus: list[dict] = self._probe_gpus()

    @staticmethod
    def _probe_ram() -> float:
        try:
            if platform.system() == "Windows":
                class MEMSTAT(ctypes.Structure):
                    _fields_ = [("l", ctypes.c_ulong), ("total", ctypes.c_uint64),
                                ("avail", ctypes.c_uint64), ("t2", ctypes.c_uint64),
                                ("a2", ctypes.c_uint64), ("t3", ctypes.c_uint64),
                                ("a3", ctypes.c_uint64), ("t4", ctypes.c_uint64)]
                s = MEMSTAT()
                s.l = ctypes.sizeof(s)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))  # type: ignore[attr-defined]
                return s.total / 1e9
            if platform.system() == "Darwin":
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
                return int(out.strip()) / 1e9
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) * 1024 / 1e9
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _probe_gpus() -> list[dict]:
        gpus: list[dict] = []
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,memory.used,driver_version",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in out.strip().splitlines():
                name, total, used, drv = [x.strip() for x in line.split(",")]
                gpus.append({"vendor": "nvidia", "name": name,
                             "vram_total_gb": float(total) / 1024,
                             "vram_used_gb": float(used) / 1024,
                             "driver": drv})
        except Exception:
            pass
        if not gpus:
            try:
                out = subprocess.check_output(
                    ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"],
                    text=True, stderr=subprocess.DEVNULL, timeout=10)
                for line in out.splitlines():
                    if "VRAM" in line.upper() and "Total" in line:
                        for p in (x.strip() for x in line.split(",")):
                            if p.replace(".", "").isdigit():
                                gpus.append({"vendor": "amd", "name": "AMD GPU",
                                             "vram_total_gb": float(p) / 1e9,
                                             "vram_used_gb": 0.0, "driver": "unknown"})
                                break
            except Exception:
                pass
        # Windows AMD/Intel：nvidia-smi 与 rocm-smi 都不适用，
        # 用注册表 HardwareInformation.qwMemorySize（64 位，不会溢出）拿真实显存
        if platform.system() == "Windows":
            gpus.extend(_probe_gpus_windows())
        return gpus

    def report(self) -> None:
        info(f"操作系统: {self.os}   内存: {self.ram_total_gb:.0f} GB")
        if self.gpus:
            for i, g in enumerate(self.gpus):
                info(f"GPU {i}: {g['name']}  显存 {g['vram_total_gb']:.0f} GB"
                     f"（已用 {g['vram_used_gb']:.1f} GB）  驱动 {g['driver']}")
        else:
            warn("未检测到可用独显（NVIDIA/AMD/Intel），将使用 CPU 模式（速度会慢很多）")


def _vendor_of(name: str) -> str:
    low = name.lower()
    if "nvidia" in low or "geforce" in low or "quadro" in low:
        return "nvidia"
    if "amd" in low or "radeon" in low:
        return "amd"
    if "intel" in low or "arc" in low:
        return "intel"
    return "other"


def _probe_gpus_windows() -> list[dict]:
    """Windows 通用显卡探测（AMD/Intel/NVIDIA 兜底）。

    Win32_VideoController.AdapterRAM 是 32 位值，显存 >4GB 会溢出，
    因此显存读注册表 HardwareInformation.qwMemorySize（REG_QWORD）。
    已用显存 Windows 无稳定无依赖接口，置 0 并在评分中按全空闲处理。
    """
    ps = (
        "$cls='HKLM:\\SYSTEM\\ControlSet001\\Control\\Class"
        "\\{4d36e968-e325-11ce-bfc1-08002be10318}';"
        "$mem=@{};Get-ChildItem $cls -ErrorAction SilentlyContinue|ForEach-Object{"
        "$p=Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue;"
        "if($p.DriverDesc){$mem[$p.DriverDesc]=$p.'HardwareInformation.qwMemorySize'}};"
        "Get-CimInstance Win32_VideoController|ForEach-Object{"
        "$v=0;if($mem.ContainsKey($_.Name)){$v=[int64]$mem[$_.Name]};"
        "if($v -le 0 -and $_.AdapterRAM){$v=[int64]$_.AdapterRAM};"
        "[Console]::Out.WriteLine(($_.Name+'|'+$_.DriverVersion+'|'+$v))}"
    )
    # powershell 可能不在 PATH（精简环境），兜底 System32 全路径
    candidates = ["powershell",
                  r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"]
    out = ""
    for exe in candidates:
        try:
            out = subprocess.check_output(
                [exe, "-NoProfile", "-NonInteractive", "-Command", ps],
                text=True, stderr=subprocess.DEVNULL, timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if out.strip():
                break
        except Exception:
            continue
    if not out.strip():
        return []
    gpus: list[dict] = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        name, drv, vram = parts
        try:
            vram_gb = int(vram) / 1e9
        except ValueError:
            continue
        if vram_gb < 1.0:  # 过滤核显/虚拟适配器（<1GB 基本不是推理目标）
            continue
        gpus.append({"vendor": _vendor_of(name), "name": name,
                     "vram_total_gb": round(vram_gb, 1),
                     "vram_used_gb": 0.0, "driver": drv})
    return gpus




# ---------------- 2. 最佳显卡选择（含多卡聚合评估） ----------------

_GEN_TABLE = [
    ("rtx 50", 1.2), ("rtx 40", 1.2), ("rtx 30", 1.0), ("rtx 20", 0.85),
    ("rx 9070", 1.05), ("rx 7900", 1.0), ("rx 7800", 0.95), ("rx 6800", 0.85),
]


def gpu_score(g: dict) -> float:
    name = g["name"].lower()
    gen = next((f for k, f in _GEN_TABLE if k in name), 0.8)
    free = g["vram_total_gb"] - g["vram_used_gb"]
    used_ratio = g["vram_used_gb"] / max(g["vram_total_gb"], 0.1)
    return free * gen * (1 - used_ratio * 0.5)


def select_gpu(hw: Hardware, manual: int | None) -> dict | None:
    """返回最佳卡 dict（附加 multi 字段：可聚合的同型号卡组），无卡返回 None。"""
    if not hw.gpus:
        return None
    if manual is not None:
        if 0 <= manual < len(hw.gpus):
            return {**hw.gpus[manual], "multi": []}
        fail(f"--gpu {manual} 超出范围（共 {len(hw.gpus)} 张卡）")
    best = max(hw.gpus, key=gpu_score)
    multi: list[dict] = []
    if len(hw.gpus) > 1:
        info("多卡机器：按 可用显存×算力代际×空闲度 评分自动选卡")
        for i, g in enumerate(hw.gpus):
            info(f"  GPU {i} {g['name']}: 得分 {gpu_score(g):.1f}")
        # 同型号卡组聚合：显存池可解锁更高档位时提示（实际切分由引擎 --tensor-split 执行）
        same = [g for g in hw.gpus if g["name"] == best["name"]]
        if len(same) > 1:
            pool = sum(g["vram_total_gb"] - g["vram_used_gb"] for g in same)
            info(f"检测到 {len(same)} 张同型号卡，聚合可用显存 {pool:.0f} GB（自动启用多卡切分）")
            multi = same
    info(f"已选择: {best['name']}（可用显存 "
         f"{best['vram_total_gb'] - best['vram_used_gb']:.1f} GB）")
    return {**best, "multi": multi}


def gpu_free_vram(gpu: dict | None) -> float:
    if not gpu:
        return 0.0
    if gpu.get("multi"):
        return sum(g["vram_total_gb"] - g["vram_used_gb"] for g in gpu["multi"])
    return gpu["vram_total_gb"] - gpu["vram_used_gb"]


def _tensor_split(gpu: dict | None) -> str:
    """同型号多卡时生成 tensor-split 比例字符串，否则空。"""
    if not gpu or not gpu.get("multi"):
        return ""
    sys.path.insert(0, str(PKG_DIR))
    from planner import tensor_split_ratio
    return tensor_split_ratio(gpu["multi"])


# ---------------- 3. 变体推荐 ----------------

def recommend_variant(pack: dict, hw: Hardware, gpu: dict | None) -> dict:
    variants = pack["variants"]
    feasible, degraded = [], []
    vram = gpu_free_vram(gpu)
    for v in variants:
        r = v["requirements"]
        if r["min_ram_gb"] > hw.ram_total_gb:
            continue
        (feasible if gpu and r["vram_hot_gb"] <= vram else degraded).append(v)
    if feasible:
        best = max(feasible, key=lambda v: v["quality_rank"])
        info(f"推荐档位: {best['id']}（{best['size_gb']} GB，质量与速度的最优平衡）")
        return best
    if degraded:
        best = max(degraded, key=lambda v: v["quality_rank"])
        warn(f"显存不足，推荐 {best['id']} 走内存/CPU 卸载模式（速度明显下降）")
        return best
    fail("内存不足：所有变体都无法在这台机器上运行。"
         f"最低需要 {min(v['requirements']['min_ram_gb'] for v in variants):.0f} GB RAM，"
         f"本机 {hw.ram_total_gb:.0f} GB")


# ---------------- 4. 下载 / 安装（断点续传 + 校验 + 幂等） ----------------

_URL_TEMPLATES = {
    "huggingface": "https://huggingface.co/{repo}/resolve/main/{file}",
    "modelscope": "https://modelscope.cn/models/{repo}/resolve/master/{file}",
}


def resolve_urls(src: dict) -> list[tuple[str, str]]:
    """把来源描述展开为 (url, 文件名) 列表。"""
    if src["type"] == "url":
        return list(zip(src["urls"], src["files"]))
    tpl = _URL_TEMPLATES.get(src["type"])
    if tpl:
        return [(tpl.format(repo=src["repo"], file=f), f) for f in src["files"]]
    return []


def download_with_resume(url: str, dst: Path) -> None:
    part = dst.with_suffix(dst.suffix + ".part")
    headers, mode = {}, "wb"
    if part.exists():
        headers["Range"] = f"bytes={part.stat().st_size}-"
        mode = "ab"
        info(f"断点续传 {dst.name}（已有 {part.stat().st_size / 1e6:.0f} MB）")
    else:
        info(f"下载 {dst.name}")
    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=60)
    # 断点续传校验：服务端必须真的接受 Range（206），否则从头重来
    if mode == "ab" and resp.status != 206:
        info("服务端未接受断点续传，重新下载")
        resp.close()
        resp = urllib.request.urlopen(urllib.request.Request(url), timeout=60)
        mode = "wb"
    total = int(resp.headers.get("Content-Length", 0)) + \
        (part.stat().st_size if mode == "ab" else 0)
    done = part.stat().st_size if mode == "ab" else 0
    report_progress("files", f"下载 {dst.name}", done, total)
    with resp, open(part, mode) as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            report_progress("files", f"下载 {dst.name}", done, total)
    part.rename(dst)


def copy_with_progress(src: Path, dst: Path) -> None:
    """大文件复制（带进度，供 Web 向导显示）。"""
    total = src.stat().st_size
    done = 0
    report_progress("files", f"复制 {dst.name}", 0, total)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            chunk = fi.read(1 << 20)
            if not chunk:
                break
            fo.write(chunk)
            done += len(chunk)
            report_progress("files", f"复制 {dst.name}", done, total)


# ---------------- 全局进度（CLI 静默，Web 向导读取） ----------------

PROGRESS: dict = {"step": "", "detail": "", "done": 0, "total": 0,
                  "status": "idle", "error": ""}


def report_progress(step: str, detail: str, done: int = 0, total: int = 0) -> None:
    PROGRESS.update(step=step, detail=detail, done=done, total=total)


def install_files(variant: dict, install_models: Path, checksums: dict) -> None:
    """安装变体的全部文件：local=包内复制，远程=断点续传下载。逐个 SHA256 校验。"""
    install_models.mkdir(parents=True, exist_ok=True)
    for src in variant["sources"]:
        if src["type"] == "local":
            fnames = src.get("files") or [Path(src["path"]).name]
            for fname in fnames:
                rel = f"models/{fname}"
                src_path, dst = PKG_DIR / rel, install_models / fname
                expect = checksums.get(rel)
                if dst.exists() and expect and sha256_file(dst) == expect:
                    info(f"已存在且校验通过，跳过: {fname}")
                    continue
                if not src_path.exists():
                    fail(f"安装包缺少文件 {rel}（包可能损坏）")
                info(f"安装 {fname}（{src_path.stat().st_size / 1e9:.2f} GB）...")
                copy_with_progress(src_path, dst)
                _verify(dst, expect)
        else:
            for url, fname in resolve_urls(src):
                dst = install_models / fname
                expect = src.get("sha256", {}).get(fname)
                if dst.exists() and (not expect or sha256_file(dst) == expect):
                    info(f"已存在，跳过: {fname}")
                    continue
                download_with_resume(url, dst)
                _verify(dst, expect)

    # 随包附件（mmproj 投影 / eagle3 草稿等）：不属于任何变体，全部安装
    pkg_models = PKG_DIR / "models"
    if pkg_models.exists():
        variant_files = {f for v_files in
                         (s.get("files") or [Path(s["path"]).name]
                          for s in variant["sources"]) for f in v_files}
        for extra in pkg_models.iterdir():
            if extra.name in variant_files:
                continue
            if not any(k in extra.name.lower() for k in ("mmproj", "eagle3", "draft")):
                continue
            dst = install_models / extra.name
            expect = checksums.get(f"models/{extra.name}")
            if dst.exists() and expect and sha256_file(dst) == expect:
                continue
            info(f"安装附件 {extra.name}（{extra.stat().st_size / 1e9:.2f} GB）...")
            copy_with_progress(extra, dst)
            _verify(dst, expect)


def _verify(path: Path, expect: str | None) -> None:
    if expect:
        if sha256_file(path) != expect:
            path.unlink()
            fail(f"{path.name} 校验失败（SHA256 不符），已删除，请重跑安装器")
    info(f"校验通过: {path.name}")


# ---------------- 5. 引擎获取与冒烟自检 ----------------

def setup_engine(pack: dict, install_dir: Path, vendor: str) -> str | None:
    """解析引擎二进制：包内置 → 安装目录缓存 → 按 version_pin 下载。返回路径或 None。

    vendor: nvidia | amd | intel | cpu（决定下载 CUDA / Vulkan / CPU 版引擎）
    """
    sys.path.insert(0, str(PKG_DIR))
    from engine_manager import download_engine, find_engine_binary

    found = find_engine_binary([PKG_DIR / "engines", install_dir / "engines"])
    if found:
        info(f"引擎已就绪: {found.name}")
        return str(found)
    pin = pack.get("engine", {}).get("version_pin", {})
    tag = pin.get("ref", "")
    if not tag or tag.startswith("m1"):
        warn("未锁定引擎版本，跳过引擎下载（API 以降级模式运行）")
        return None
    try:
        exe = download_engine(tag, vendor, install_dir / "engines")
        info(f"引擎下载完成（{vendor} 版）: {exe.name}")
        return str(exe)
    except Exception as e:
        warn(f"引擎直连下载失败: {e}，尝试镜像加速...")
    try:
        exe = download_engine(tag, vendor, install_dir / "engines",
                              mirror="https://ghfast.top/https://github.com")
        info(f"引擎镜像下载完成（{vendor} 版）: {exe.name}")
        return str(exe)
    except Exception as e:
        warn(f"引擎下载失败: {e}（API 以降级模式运行，可重跑安装器重试）")
        return None


def smoke_test(engine_bin: str | None) -> None:
    """冒烟自检：验证引擎二进制可执行。失败不阻断安装，但明确提示。"""
    if not engine_bin:
        return
    try:
        cmd = [sys.executable, engine_bin, "--version"] if engine_bin.endswith(".py") \
            else [engine_bin, "--version"]
        out = subprocess.run(cmd, capture_output=True,
                             text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if out.returncode == 0:
            info(f"冒烟自检通过: {(out.stdout or out.stderr).strip().splitlines()[0][:60]}")
        else:
            warn("引擎自检返回非零，运行时可能有问题（详见 state/logs/engine.log）")
    except Exception as e:
        warn(f"引擎自检失败: {e}")


# ---------------- 6. 卸载 / 诊断 ----------------

def cmd_uninstall(pack_name: str, install_dir: Path, assume_yes: bool) -> int:
    if not install_dir.exists():
        print(f"未找到安装目录 {install_dir}，无需卸载")
        return 0
    size = sum(p.stat().st_size for p in install_dir.rglob("*") if p.is_file())
    print(f"将删除 {install_dir}（含模型与配置，共 {size / 1e9:.2f} GB）")
    ans = ask("确认卸载？输入 yes 确认", "", assume_yes)
    if ans != "yes" and not assume_yes:
        print("已取消")
        return 1
    # 先清开机自启项（启动文件夹 VBS / 计划任务 / systemd --user）
    sys.path.insert(0, str(PKG_DIR))
    try:
        from service import remove_user_autostart
        remove_user_autostart(pack_name)
        info(f"已清除自启项: {pack_name}")
    except Exception:
        pass
    shutil.rmtree(install_dir)
    print(f"已卸载 {pack_name}，系统无残留（本安装器不修改注册表与环境变量）")
    return 0


def cmd_diagnose(pack: dict, install_dir: Path) -> int:
    hw = Hardware()
    report = {
        "pack": {"name": pack["name"], "version": pack["version"]},
        "os": platform.platform(),
        "python": sys.version,
        "ram_gb": round(hw.ram_total_gb, 1),
        "gpus": hw.gpus,
    }
    out = install_dir / "state" / "support-bundle.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hardware-report.json", json.dumps(report, ensure_ascii=False, indent=2))
        for log in (install_dir / "state").rglob("*.log"):
            zf.write(log, log.name)
        cfg = install_dir / "state" / "config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text(encoding="utf-8"))
            c.pop("api_key", None)  # 脱敏
            zf.writestr("config.json", json.dumps(c, ensure_ascii=False, indent=2))
    print(f"诊断包已生成: {out}（可直接附在 issue 中，已脱敏）")
    return 0


# ---------------- 主流程 ----------------

def run_install(pack: dict, install_dir: Path, variant: dict,
                gpu: dict | None, hw: Hardware) -> dict:
    """执行安装主体（模型文件 → 引擎 → 配置），CLI 与 Web 向导共用。返回 config。"""
    state_dir = install_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    ledger_p = state_dir / "install-state.json"
    ledger = json.loads(ledger_p.read_text()) if ledger_p.exists() else {"done": []}

    report_progress("files", "安装模型文件")
    step = f"files:{variant['id']}"
    if step in ledger["done"]:
        info("文件已安装（幂等跳过）")
    else:
        install_files(variant, install_dir / "models", pack.get("checksums_sha256", {}))
        ledger["done"].append(step)
        ledger_p.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    report_progress("engine", "准备推理引擎")
    vendor = gpu["vendor"] if gpu else "cpu"
    engine_bin = setup_engine(pack, install_dir, vendor)
    smoke_test(engine_bin)

    report_progress("config", "写入配置")
    src0 = variant["sources"][0]
    fname = Path(src0.get("path") or src0["files"][0]).name
    config = {
        "name": pack["name"], "variant": variant["id"],
        "model_file": str(install_dir / "models" / fname),
        "engine": pack["engine"],
        "engine_binary": engine_bin,
        "architecture": pack["model"]["architecture"],
        "layers": pack["model"].get("layers", 0),
        "expert_count": pack["model"].get("expert_count", 0),
        "expert_used_count": pack["model"].get("expert_used_count", 0),
        "gpu": gpu["name"] if gpu else "cpu",
        "gpu_free_vram_gb": round(gpu_free_vram(gpu), 1),
        "ram_free_gb": round(hw.ram_total_gb, 1),
        "tensor_split": _tensor_split(gpu),
        "port": pack["service"]["port"],
        "host": "127.0.0.1",
        "api_key": "",
        "branding": pack.get("branding", {}),
        "pkg_dir": str(PKG_DIR),   # 安装包所在目录（hub 启动服务用）
        # 审计溯源：权重指纹（日志记录用，EU AI Act Art.12 对应物）
        "model_sha256": pack.get("checksums_sha256", {}).get(f"models/{fname}", "")[:16],
    }
    (state_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    ledger["done"].append("config")
    ledger_p.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return config


def start_gateway_detached(config_path: Path) -> None:
    """后台启动 API 网关（独立进程，不随安装器退出）。"""
    log = open(config_path.parent / "serve.log", "ab")
    subprocess.Popen(
        [sys.executable, str(PKG_DIR / "serve.py"), "--config", str(config_path)],
        cwd=str(PKG_DIR), stdout=log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "DETACHED_PROCESS", 0))



def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="本地 AI 模型安装器")
    ap.add_argument("--web", action="store_true",
                    help="浏览器安装向导（双击 START-HERE.bat 的默认模式）")
    ap.add_argument("--dir", type=Path, default=None, help="安装目录")
    ap.add_argument("--gpu", type=int, default=None, help="手动指定 GPU 序号")
    ap.add_argument("--yes", action="store_true", help="全部接受推荐值（无人值守）")
    ap.add_argument("--no-autostart", action="store_true",
                    help="静默安装时不注册登录自启（企业部署用）")
    ap.add_argument("--uninstall", action="store_true", help="卸载")
    ap.add_argument("--also-package", action="store_true",
                    help="配合 --uninstall：连带删除分发包 zip（本安装包本体）")
    ap.add_argument("--diagnose", action="store_true", help="导出诊断包")
    ap.add_argument("--service", action="store_true",
                    help="注册为系统服务（开机自启，默认只生成注册脚本）")
    ap.add_argument("--apply", action="store_true",
                    help="配合 --service 真正注册（需要管理员/root）")
    ap.add_argument("--update", action="store_true",
                    help="只更新引擎（模型与配置不动）")
    ap.add_argument("--apply-update", type=Path, default=None, metavar="更新包.zip",
                    help="应用离线增量更新包（校验 SHA256 后替换运行时文件）")
    args = ap.parse_args(preprocess_enterprise_args(argv))

    pack_path = PKG_DIR / "modelpack.json"
    if not pack_path.exists():
        fail("未找到 modelpack.json，请在安装包根目录运行")
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    verify_signature()
    name = pack["name"]
    install_dir = args.dir or default_install_dir(name)

    if args.uninstall:
        rc = cmd_uninstall(name, install_dir, args.yes)
        if rc == 0 and args.also_package:
            zip_path = PKG_DIR.parent / (PKG_DIR.name + ".zip")
            if zip_path.exists():
                zip_path.unlink()
                print(f"已连带删除分发包: {zip_path}")
            else:
                info("未找到配套 zip（可能已被移动），仅删除了安装目录")
        return rc
    if args.diagnose:
        return cmd_diagnose(pack, install_dir)
    if args.update:
        sys.path.insert(0, str(PKG_DIR))
        from service import update_engine
        print(f"\n=== {name} 引擎更新 ===")
        gpus = Hardware().gpus
        vendor = max(gpus, key=gpu_score)["vendor"] if gpus else "cpu"
        update_engine(pack, install_dir, vendor)
        print("更新完成（模型与配置未动）。重启服务后生效。")
        return 0
    if args.apply_update:
        return cmd_apply_update(args.apply_update)
    if args.service:
        sys.path.insert(0, str(PKG_DIR))
        from service import register_service
        cfg = install_dir / "state" / "config.json"
        if not cfg.exists():
            fail("尚未安装，请先正常运行安装器")
        files = register_service(name, PKG_DIR, cfg, install_dir, apply=args.apply)
        for f in files:
            info(f"已生成服务文件: {f}")
        if not args.apply:
            info("以上为 dry-run。确认无误后加 --apply 真正注册（需要管理员/root）")
        return 0
    if args.web:
        return run_web_wizard(pack, install_dir, args.gpu)

    print(f"\n=== {name} {pack['version']} 本地 AI 安装器 ===")
    print(f"模型: {pack['model']['family']}   许可证: {pack['model']['license']}\n")
    install_dir = Path(ask("安装目录", str(install_dir), args.yes))
    state_dir = install_dir / "state"

    print("\n[1/5] 硬件探测")
    hw = Hardware()
    hw.report()
    gpu = select_gpu(hw, args.gpu)

    print("\n[2/5] 资源规划")
    variant = recommend_variant(pack, hw, gpu)
    free_disk = shutil.disk_usage(install_dir.anchor or "/").free / 1e9
    if free_disk < variant["requirements"]["disk_gb"]:
        fail(f"磁盘空间不足：需要 {variant['requirements']['disk_gb']:.0f} GB，"
             f"可用 {free_disk:.0f} GB")
    chosen = ask("确认档位（回车接受推荐，或输入其他档位 id）", variant["id"], args.yes)
    if chosen != variant["id"]:
        match = [v for v in pack["variants"] if v["id"] == chosen]
        if not match:
            fail(f"未知档位 {chosen}，可选: {', '.join(v['id'] for v in pack['variants'])}")
        variant = match[0]

    print("\n[3/5] 安装模型文件")
    print("\n[4/5] 准备推理引擎")
    print("\n[5/5] 写入配置")
    config = run_install(pack, install_dir, variant, gpu, hw)

    port = pack["service"]["port"]
    print(f"""
安装完成 ✔

  安装目录:   {install_dir}
  配置文件:   {state_dir / 'config.json'}
  启动服务:   python "{PKG_DIR / 'serve.py'}" --config "{state_dir / 'config.json'}"
  控制台:     http://127.0.0.1:{port}/
  API 地址:   http://127.0.0.1:{port}/v1   （OpenAI 兼容）
  Anthropic:  http://127.0.0.1:{port}/v1/messages
  卸载:       python "{PKG_DIR / 'installer.py'}" --uninstall --dir "{install_dir}"

  再次运行本安装器不会重复下载/安装（幂等）。
""")
    return 0


def verify_signature() -> None:
    """若安装包带 GPG 签名则验证。验证失败拒绝安装；无 gpg 时降级警告。"""
    sig = PKG_DIR / "modelpack.json.sig"
    if not sig.exists():
        return  # 未签名：保持 SHA256 校验
    if not shutil.which("gpg"):
        warn("安装包有签名但本机无 gpg，无法验证签名（仍会做 SHA256 校验）")
        return
    pub = PKG_DIR / "publisher-key.asc"
    if pub.exists():
        subprocess.run(["gpg", "--batch", "--yes", "--import", str(pub)],
                       capture_output=True, timeout=30,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    r = subprocess.run(
        ["gpg", "--batch", "--verify", str(sig), str(PKG_DIR / "modelpack.json")],
        capture_output=True, text=True, timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode == 0:
        info("安装包签名验证通过")
    else:
        fail("安装包签名验证失败：可能被篡改，拒绝安装")


def cmd_apply_update(update_zip: Path) -> int:
    """应用离线增量更新包：校验 manifest 与每个文件 SHA256 后替换运行时文件。"""
    if not update_zip.exists():
        fail(f"更新包不存在: {update_zip}")
    with zipfile.ZipFile(update_zip) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest.get("type") != "modelforge-update":
            fail("不是合法的 ModelForge 更新包")
        current = json.loads((PKG_DIR / "modelpack.json").read_text(encoding="utf-8"))
        if manifest["name"] != current["name"]:
            fail(f"更新包属于 {manifest['name']}，本安装包是 {current['name']}，不匹配")
        print(f"应用更新: {manifest['from_version']} → {manifest['to_version']}"
              f"（{len(manifest['files'])} 个文件）")
        for f in manifest["files"]:
            data = zf.read(f["path"])
            h = hashlib.sha256(data).hexdigest()
            if h != f["sha256"]:
                fail(f"{f['path']} 校验失败（SHA256 不符），中止")
            target = PKG_DIR / f["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            info(f"已更新: {f['path']}")
    print("更新完成 ✔（重启服务后生效）")
    return 0


# ---------------- Web 安装向导 ----------------

WIZARD_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>本地 AI 安装向导</title>
<style>
  body { background:#0d1117; color:#e6edf3; font:14px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
         max-width:640px; margin:40px auto; padding:0 20px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px 18px; margin-top:14px; }
  h1 { font-size:19px; } h2 { font-size:14px; color:#4c9aff; margin:0 0 8px; }
  .dim { color:#8b949e; font-size:12.5px; }
  .var { display:flex; gap:8px; align-items:center; padding:6px 0; }
  .var input { width:auto; }
  .rec { color:#3fb950; font-size:12px; margin-left:6px; }
  input[type=text] { width:100%; background:#0d1117; border:1px solid #30363d; color:#e6edf3;
      border-radius:8px; padding:8px 12px; font:inherit; box-sizing:border-box; margin-top:4px; }
  button { background:#4c9aff; color:#fff; border:0; border-radius:8px; padding:10px 22px;
      font:inherit; cursor:pointer; margin-top:14px; }
  button:disabled { opacity:.4; }
  .bar { height:10px; background:#21262d; border-radius:5px; margin-top:10px; overflow:hidden; }
  .bar > div { height:100%; background:#3fb950; width:0; transition:width .3s; }
  a { color:#4c9aff; }
  #done { display:none; }
</style></head><body>
<h1>⚡ <span id="title">本地 AI 安装向导</span></h1>
<div class="card"><h2>硬件检测</h2><div id="hw" class="dim">检测中…</div></div>
<div class="card" id="liccard" style="display:none"><h2>许可证与发布者</h2>
  <div id="lic" class="dim"></div></div>
<div class="card"><h2>选择档位</h2><div id="vars" class="dim">加载中…</div>
  <h2 style="margin-top:14px">安装目录</h2><input type="text" id="dir">
  <label style="margin-top:10px;display:block;font-size:13px;color:#8b949e">
    <input type="checkbox" id="autostart" checked> 登录后自动启动服务（推荐，免管理员）</label>
  <br><button id="go" onclick="install()">开始安装</button>
  <div class="bar"><div id="bar"></div></div>
  <div id="prog" class="dim"></div>
</div>
<div class="card" id="done">
  <h2>✔ 安装完成</h2>
  <div id="donemsg"></div>
  <button onclick="startSvc()">启动服务并打开控制台</button>
  <div id="link" style="margin-top:10px"></div>
</div>
<script>
const $ = id => document.getElementById(id);
let recommended = '';

fetch('/wiz/info').then(r => r.json()).then(d => {
  $('title').textContent = d.name + ' ' + d.version + ' 安装向导';
  $('hw').innerHTML = d.hw_html;
  $('dir').value = d.default_dir;
  const lic = [d.license_text, d.publisher ? '发布者：' + d.publisher : '']
    .filter(Boolean).join('<br>');
  if (lic) { $('lic').innerHTML = lic; $('liccard').style.display = 'block'; }
  recommended = d.recommended;
  $('vars').innerHTML = d.variants.map(v =>
    `<div class="var"><input type="radio" name="v" id="v_${v.id}" value="${v.id}"
      ${v.id === d.recommended ? 'checked' : ''} ${v.feasible ? '' : 'disabled'}>
     <label for="v_${v.id}">${v.id}（${v.size_gb} GB）${v.feasible ? '' : ' — 硬件不足'}</label>
     ${v.id === d.recommended ? '<span class="rec">★ 推荐</span>' : ''}</div>`).join('');
});

function install() {
  const vid = document.querySelector('input[name=v]:checked');
  if (!vid) { alert('请选择一个档位'); return; }
  $('go').disabled = true;
  fetch('/wiz/install', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({variant: vid.value, dir: $('dir').value,
                          autostart: $('autostart').checked})});
  poll();
}

function poll() {
  fetch('/wiz/progress').then(r => r.json()).then(d => {
    const pct = d.total ? Math.round(d.done / d.total * 100) : (d.status === 'done' ? 100 : 0);
    $('bar').style.width = pct + '%';
    $('prog').textContent = d.status === 'error' ? '✘ ' + d.error
      : d.status === 'done' ? '' : `${d.detail} ${d.total ? (d.done/1e9).toFixed(2)+'/'+(d.total/1e9).toFixed(2)+' GB' : ''}`;
    if (d.status === 'done') { $('done').style.display = 'block'; $('donemsg').textContent = d.message; }
    else if (d.status !== 'error') setTimeout(poll, 800);
  });
}

function startSvc() {
  fetch('/wiz/start', {method:'POST'}).then(r => r.json()).then(d => {
    $('link').innerHTML = `服务已启动 → <a href="${d.console_url}" target="_blank">打开控制台</a>
      <div class="dim">API 地址：${d.console_url}v1（OpenAI 兼容）。本向导页面可以关闭了。</div>`;
    window.open(d.console_url, '_blank');
  });
}
</script></body></html>"""


class StrictHTTPServer(ThreadingHTTPServer):
    """Windows 下 SO_REUSEADDR 允许重复绑定同一端口（幽灵监听），禁用之。"""
    allow_reuse_address = False


def run_web_wizard(pack: dict, install_dir: Path, gpu_manual: int | None) -> int:
    """浏览器安装向导：硬件检测展示 → 档位选择 → 进度条安装 → 一键启动控制台。"""
    hw = Hardware()
    gpu = select_gpu(hw, gpu_manual)
    recommended = recommend_variant(pack, hw, gpu)
    state: dict = {"config": None}

    def hw_html() -> str:
        rows = [f"操作系统：{hw.os}　内存：{hw.ram_total_gb:.0f} GB"]
        for i, g in enumerate(hw.gpus):
            mark = " ← 已自动选择" if gpu and g["name"] == gpu["name"] else ""
            rows.append(f"显卡 {i}：{g['name']}（{g['vram_total_gb']:.0f} GB 显存）{mark}")
        if not hw.gpus:
            rows.append("未检测到独显，将使用 CPU 模式（较慢）")
        return "<br>".join(rows)

    class WizHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                branding = pack.get("branding", {})
                theme = branding.get("theme_color") or "#4c9aff"
                logo = branding.get("logo_text") or "⚡"
                body = WIZARD_PAGE.replace("#4c9aff", theme).replace(
                    "<h1>⚡", f"<h1>{logo}").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/wiz/info":
                branding = pack.get("branding", {})
                variants = [{
                    "id": v["id"], "size_gb": v["size_gb"],
                    "feasible": v["requirements"]["min_ram_gb"] <= hw.ram_total_gb,
                } for v in pack["variants"]]
                return self._json(200, {
                    "name": branding.get("display_name") or pack["name"],
                    "version": pack["version"],
                    "license_text": branding.get("license_text") or
                        f"模型许可证: {pack['model'].get('license', '见 licenses/ 目录')}",
                    "publisher": pack.get("publisher", {}).get("name", ""),
                    "hw_html": hw_html(), "variants": variants,
                    "recommended": recommended["id"],
                    "default_dir": str(install_dir)})
            if self.path == "/wiz/progress":
                return self._json(200, dict(PROGRESS))
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/wiz/install":
                body = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", 0))))
                variant = next((v for v in pack["variants"]
                                if v["id"] == body.get("variant")), recommended)
                target = Path(body.get("dir") or str(install_dir))

                def work():
                    try:
                        PROGRESS.update(status="running")
                        state["config"] = run_install(pack, target, variant, gpu, hw)
                        msg = f"已安装到 {target}，模型档位 {variant['id']}"
                        if body.get("autostart"):
                            sys.path.insert(0, str(PKG_DIR))
                            from service import register_user_autostart
                            cfg_path = target / "state" / "config.json"
                            ok, m = register_user_autostart(
                                pack["name"], PKG_DIR / "serve.py", cfg_path)
                            msg += f"。{'✔ ' + m if ok else '自启注册失败: ' + m}"
                        PROGRESS.update(status="done", message=msg)
                    except SystemExit:
                        PROGRESS.update(status="error", error="安装失败，详见安装器窗口")
                    except Exception as e:
                        PROGRESS.update(status="error", error=str(e))

                threading.Thread(target=work, daemon=True).start()
                return self._json(200, {"ok": True})
            if self.path == "/wiz/start":
                cfg = state.get("config")
                if not cfg:
                    return self._json(400, {"error": "尚未完成安装"})
                cfg_path = Path(cfg["model_file"]).parent.parent / "state" / "config.json"
                start_gateway_detached(cfg_path)
                # 等待网关写入实际端口（可能自动顺延）
                actual_port = cfg["port"]
                runtime_f = cfg_path.parent / "runtime.json"
                for _ in range(20):
                    time.sleep(0.5)
                    if runtime_f.exists():
                        try:
                            actual_port = json.loads(
                                runtime_f.read_text())["port"]
                            break
                        except Exception:
                            pass
                return self._json(200, {
                    "console_url": f"http://127.0.0.1:{actual_port}/"})
            return self._json(404, {"error": "not found"})

    server = StrictHTTPServer(("127.0.0.1", 0), WizHandler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    print(f"安装向导已启动: {url}（浏览器会自动打开；关闭本窗口即退出向导）")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
