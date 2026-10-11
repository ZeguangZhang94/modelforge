#!/usr/bin/env python3
"""安装包 GPG 签名与验证（可选能力，检测到 gpg 才启用）。

- 打包者：build --sign → 对 modelpack.json 生成 modelpack.json.sig 分离签名，
  并把公钥导出为 publisher-key.asc 随包分发
- 安装器：若包内有 .sig 且有 gpg → 验证签名，失败则拒绝安装；
  无 gpg 时明确警告"无法验证签名"
- 无签名的包保持现有 SHA256 校验行为（向后兼容）
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def gpg_available() -> bool:
    return shutil.which("gpg") is not None


def sign_package(root: Path) -> Path | None:
    """对 modelpack.json 做分离签名，并导出公钥。无 gpg 返回 None。"""
    if not gpg_available():
        return None
    mp = root / "modelpack.json"
    sig = root / "modelpack.json.sig"
    r = subprocess.run(
        ["gpg", "--batch", "--yes", "--detach-sign", "--armor",
         "--output", str(sig), str(mp)],
        capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not sig.exists():
        return None
    # 导出签名者公钥随包分发
    keyid = subprocess.run(
        ["gpg", "--batch", "--list-packets", str(sig)],
        capture_output=True, text=True, timeout=30)
    key = ""
    for line in keyid.stdout.splitlines():
        if "keyid" in line:
            key = line.strip().split()[-1]
            break
    if key:
        pub = subprocess.run(["gpg", "--batch", "--yes", "--armor", "--export", key],
                             capture_output=True, text=True, timeout=30)
        if pub.returncode == 0 and pub.stdout:
            (root / "publisher-key.asc").write_text(pub.stdout, encoding="utf-8")
    return sig


def verify_package(root: Path) -> tuple[bool, str]:
    """验证包签名。返回 (可否继续, 消息)。"""
    mp = root / "modelpack.json"
    sig = root / "modelpack.json.sig"
    if not sig.exists():
        return True, "包未签名（仅 SHA256 校验）"
    if not gpg_available():
        return True, "警告：包有签名但本机无 gpg，无法验证（仍做 SHA256 校验）"
    pub = root / "publisher-key.asc"
    if pub.exists():
        subprocess.run(["gpg", "--batch", "--yes", "--import", str(pub)],
                       capture_output=True, timeout=30)
    r = subprocess.run(["gpg", "--batch", "--verify", str(sig), str(mp)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode == 0:
        return True, "签名验证通过"
    return False, "签名验证失败：安装包可能被篡改，拒绝安装"
