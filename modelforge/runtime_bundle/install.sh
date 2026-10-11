#!/usr/bin/env sh
# ModelForge Linux 一键入口：默认打开浏览器安装向导
set -e
cd "$(dirname "$0")"
PY=""
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else
    echo "[错误] 未找到 Python，请先安装 Python 3.9+（如 sudo apt install python3）" >&2
    exit 1
fi
if [ $# -eq 0 ]; then
    exec $PY installer.py --web
else
    exec $PY installer.py "$@"
fi
