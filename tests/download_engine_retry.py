"""带重试的下载循环：断点续传 + 最多 N 次尝试，用于不稳定镜像网络。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "modelforge" / "runtime_bundle"))
from engine_manager import download_engine

TARGET = Path(__file__).parent / "engine_cache"
ATTEMPTS = 8

for i in range(1, ATTEMPTS + 1):
    try:
        exe = download_engine("b11526", "amd", TARGET,
                              mirror="https://ghfast.top/https://github.com")
        print(f"第 {i} 次尝试成功: {exe}")
        sys.exit(0)
    except Exception as e:
        parts = list(TARGET.glob("*.part"))
        done = parts[0].stat().st_size / 1e6 if parts else 0
        print(f"第 {i} 次中断: {type(e).__name__}  已下载 {done:.1f}/33.5 MB")
        time.sleep(3)

print("重试次数耗尽")
sys.exit(1)
