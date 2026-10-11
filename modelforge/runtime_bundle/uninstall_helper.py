#!/usr/bin/env python3
"""自卸载辅助进程：由控制台「卸载」按钮触发，独立进程运行。

流程：杀网关与引擎进程 → 等待句柄释放 → 清理开机自启项 → 删除安装目录。
必须独立进程运行（安装目录删除时网关自身也随之消失）。
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def kill_pid(pid: int, tree: bool = False) -> None:
    """杀进程。tree=True 才杀进程树——网关不能用树杀（会把辅助进程一起带走）。"""
    try:
        if platform.system() == "Windows":
            cmd = ["taskkill", "/PID", str(pid), "/F"] + (["/T"] if tree else [])
            subprocess.run(cmd, capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.kill(pid, 15)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--engine-pid", type=int, default=0)
    ap.add_argument("--gateway-pid", type=int, default=0)
    ap.add_argument("--marker", type=Path, default=None,
                    help="完成后写标记文件（调用方校验用）")
    args = ap.parse_args()

    # 顺序：先引擎（带进程树），后网关（不带树，保住本辅助进程）
    if args.engine_pid:
        kill_pid(args.engine_pid, tree=True)
    if args.gateway_pid:
        kill_pid(args.gateway_pid, tree=False)

    # 等进程退出、文件句柄释放（Windows 上运行中的 exe/映射文件删不掉）
    time.sleep(3)

    # 清理开机自启（启动文件夹 VBS / 计划任务 / systemd --user）
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from service import remove_user_autostart
        remove_user_autostart(args.name)
    except Exception:
        pass

    # 删除安装目录（重试 3 次，等杀软/索引释放句柄）
    for _ in range(3):
        try:
            if args.dir.exists():
                shutil.rmtree(args.dir)
            break
        except OSError:
            time.sleep(2)

    if args.marker:
        ok = not args.dir.exists()
        args.marker.write_text(json.dumps({
            "uninstalled": ok, "dir": str(args.dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
