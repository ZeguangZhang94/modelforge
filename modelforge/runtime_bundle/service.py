#!/usr/bin/env python3
"""系统服务注册与更新机制（M4）。

- register_service：生成并（可选）注册系统服务，开机自启 API 网关
    Windows → sc.exe 服务（包装 bat）；Linux → systemd unit；macOS → launchd plist
    默认只生成文件（dry-run），--apply 才真正注册（需要管理员/root）
- update_engine：引擎与模型分离更新——只拉新引擎二进制，权重与配置不动

纯标准库。文件生成逻辑可单元测试，实际注册需要提权，由用户显式确认。
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


# ---------------- 服务文件生成（纯函数，可测） ----------------

def gen_systemd_unit(name: str, serve_py: Path, config: Path, workdir: Path) -> str:
    return f"""[Unit]
Description={name} local AI API
After=network.target

[Service]
Type=simple
WorkingDirectory={workdir}
ExecStart={sys.executable} {serve_py} --config {config}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
"""


def gen_launchd_plist(name: str, serve_py: Path, config: Path, workdir: Path) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>local.aiinstaller.{name}</string>
  <key>ProgramArguments</key><array>
    <string>{sys.executable}</string>
    <string>{serve_py}</string>
    <string>--config</string><string>{config}</string>
  </array>
  <key>WorkingDirectory</key><string>{workdir}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict></plist>
"""


def gen_windows_service_bat(name: str, serve_py: Path, config: Path, workdir: Path) -> str:
    py = str(Path(sys.executable))
    # 用户级登录任务：/sc onlogon 对当前用户无需管理员权限
    return f"""@echo off
REM 注册 {name} 登录自启（任务计划，当前用户级别，无需管理员）
schtasks /create /tn "{name}" /sc onlogon /tr "\\"{py}\\" \\"{serve_py}\\" --config \\"{config}\\"" /f
schtasks /run /tn "{name}"
echo 已注册并启动。卸载: schtasks /delete /tn "{name}" /f
"""


def _startup_vbs(name: str, serve_py: Path, config: Path) -> Path:
    """生成隐藏启动的 VBS 到用户启动文件夹（完全免权限）。"""
    startup = Path(os.environ.get("APPDATA", str(Path.home()))) / \
        "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    startup.mkdir(parents=True, exist_ok=True)
    py = str(Path(sys.executable))
    vbs = startup / f"{name}.vbs"
    vbs.write_text(
        f'CreateObject("Wscript.Shell").Run """{py}"" ""{serve_py}"" --config ""{config}""", 0, False\n',
        encoding="utf-8")
    return vbs


def remove_user_autostart(name: str) -> None:
    """清除自启（启动文件夹 VBS + 计划任务双保险）。"""
    if platform.system() == "Windows":
        startup = Path(os.environ.get("APPDATA", str(Path.home()))) / \
            "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / f"{name}.vbs"
        startup.unlink(missing_ok=True)
        import subprocess as sp
        sp.run(["schtasks", "/delete", "/tn", name, "/f"], capture_output=True,
               creationflags=getattr(sp, "CREATE_NO_WINDOW", 0))
    elif platform.system() == "Linux":
        import subprocess as sp
        sp.run(["systemctl", "--user", "disable", "--now", name], capture_output=True)
        (Path.home() / ".config" / "systemd" / "user" / f"{name}.service").unlink(
            missing_ok=True)


def register_user_autostart(name: str, serve_py: Path, config: Path) -> tuple[bool, str]:
    """注册当前用户登录自启。Windows 用启动文件夹 VBS（零权限）；
    Linux 用 systemd --user。返回 (是否成功, 消息)。
    """
    import subprocess as sp
    if platform.system() == "Windows":
        try:
            vbs = _startup_vbs(name, serve_py, config)
            if vbs.exists():
                return True, "已注册登录自启（下次登录自动启动服务）"
            return False, "启动文件夹写入失败"
        except Exception as e:
            return False, str(e)[:200]
    if platform.system() == "Linux":
        unit_dir = Path.home() / ".config" / "systemd" / "user"
        unit_dir.mkdir(parents=True, exist_ok=True)
        (unit_dir / f"{name}.service").write_text(
            gen_systemd_unit(name, serve_py, config, serve_py.parent)
            .replace("multi-user.target", "default.target"), encoding="utf-8")
        r = sp.run(["systemctl", "--user", "enable", "--now", name],
                   capture_output=True, text=True, timeout=30)
        return (r.returncode == 0,
                "已注册用户级 systemd 服务" if r.returncode == 0 else r.stderr[:200])
    return False, "当前系统不支持自动注册"


# ---------------- 注册入口 ----------------

def register_service(name: str, pkg_dir: Path, config: Path,
                     install_dir: Path, apply: bool = False) -> list[Path]:
    """生成服务文件。apply=True 时尝试真正注册（需要权限）。"""
    serve_py = pkg_dir / "serve.py"
    out_dir = install_dir / "state" / "service"
    out_dir.mkdir(parents=True, exist_ok=True)
    system = platform.system()
    written: list[Path] = []

    if system == "Windows":
        f = out_dir / "register-service.bat"
        f.write_text(gen_windows_service_bat(name, serve_py, config, pkg_dir),
                     encoding="utf-8")
        written.append(f)
        if apply:
            import subprocess
            subprocess.run(["cmd", "/c", str(f)], check=True)
    elif system == "Linux":
        f = out_dir / f"{name}.service"
        f.write_text(gen_systemd_unit(name, serve_py, config, pkg_dir),
                     encoding="utf-8")
        written.append(f)
        if apply:
            import subprocess
            subprocess.run(["sudo", "cp", str(f), "/etc/systemd/system/"], check=True)
            subprocess.run(["sudo", "systemctl", "enable", "--now", name], check=True)
    elif system == "Darwin":
        f = out_dir / f"local.aiinstaller.{name}.plist"
        f.write_text(gen_launchd_plist(name, serve_py, config, pkg_dir),
                     encoding="utf-8")
        written.append(f)
        if apply:
            import subprocess
            dst = Path.home() / "Library" / "LaunchAgents" / f.name
            dst.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
            subprocess.run(["launchctl", "load", str(dst)], check=True)
    return written


# ---------------- 更新机制（引擎与模型分离） ----------------

def update_engine(pack: dict, install_dir: Path, vendor: str) -> str | None:
    """只更新引擎：清理旧引擎目录后按 version_pin 重下。模型权重与配置不动。"""
    import shutil
    sys.path.insert(0, str(Path(__file__).parent))
    from engine_manager import download_engine

    engines = install_dir / "engines"
    if engines.exists():
        shutil.rmtree(engines)
    pin = pack.get("engine", {}).get("version_pin", {})
    tag = pin.get("ref", "")
    if not tag:
        print("  未锁定引擎版本，无可更新内容")
        return None
    exe = download_engine(tag, vendor, engines)
    print(f"  引擎已更新到 {tag}: {exe.name}")
    return str(exe)
