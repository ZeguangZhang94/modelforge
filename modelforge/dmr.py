#!/usr/bin/env python3
"""Linux Docker Model Runner（DMR）打包路径。

产品决策：Linux 分发走 Docker 生态（OCI 制品 + `docker model pull`），
Windows 维持 ModelForge 自包含安装包。

DMR 安装包内容：
    <name>-dmr/
    ├── install.sh          # 检测 docker/插件 → 拉模型 → 打印 API 地址
    ├── docker-compose.yml  # 给应用栈引用的 models: 声明示例
    └── README.md           # 用法与排障

模型引用优先级：DMR 官方 Hub 命名空间 > HuggingFace GGUF 仓库（hf.co 前缀，DMR 自动转 OCI）。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

INSTALL_SH = """#!/usr/bin/env bash
# {display_name} - Docker Model Runner 安装器（Linux）
set -e

echo "=== {display_name} 安装 ==="

# 1. 检测 Docker
if ! command -v docker >/dev/null 2>&1; then
    echo "[错误] 未找到 Docker。安装：https://docs.docker.com/engine/install/"
    exit 1
fi

# 2. 检测 Docker Model Runner 插件
if ! docker model version >/dev/null 2>&1; then
    echo "安装 docker-model-plugin..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y docker-model-plugin
    else
        echo "[错误] 请手动安装 docker-model-plugin（或升级 Docker Desktop 4.40+）"
        exit 1
    fi
fi

# 3. 拉取模型（DMR 自动把 HF 的 GGUF 转为 OCI 制品并缓存）
echo "拉取模型 {model_ref} ..."
docker model pull {model_ref}

# 4. 完成
echo ""
echo "安装完成 ✔"
echo "  命令行对话:  docker model run {model_ref} \\"你好\\""
echo "  OpenAI API:  http://localhost:12434/v1  （model 名: {model_ref}）"
echo "  卸载:        docker model rm {model_ref}"
"""

COMPOSE = """# 在你的应用栈中引用本模型（Docker Compose 方式）
services:
  app:
    image: your-app:latest
    models:
      - {service_name}

models:
  {service_name}:
    model: {model_ref}
"""

README = """# {display_name}（Docker Model Runner 版）

适用：Linux（需 Docker + docker-model-plugin，或 Docker Desktop 4.40+）。

## 安装
```bash
./install.sh
```

## 使用
- 命令行对话：`docker model run {model_ref} "你好"`
- OpenAI 兼容 API：`http://localhost:12434/v1`（宿主机）
- 容器内访问：`http://model-runner.docker.internal/v1`
- 应用栈集成：参考 docker-compose.yml 的 `models:` 声明

## 卸载
```bash
docker model rm {model_ref}
```

## 排障
- `docker model version` 报错 → 装插件：`sudo apt-get install docker-model-plugin`
- GPU 未启用 → DMR 自动检测 CUDA；`docker model status` 查看后端
- 更多：https://docs.docker.com/ai/model-runner/
"""


def build_dmr_package(name: str, model_ref: str, out_dir: str | Path,
                      display_name: str = "") -> Path:
    """生成 DMR 安装包目录 + zip。返回 zip 路径。

    model_ref 示例： hf.co/unsloth/Qwen3.8-27B-GGUF  或  ai/qwen3 等 Hub 官方名
    """
    display_name = display_name or name
    service_name = name.replace("-", "_")
    root = Path(out_dir) / f"{name}-dmr"
    root.mkdir(parents=True, exist_ok=True)
    ctx = {"display_name": display_name, "model_ref": model_ref,
           "service_name": service_name}
    (root / "install.sh").write_text(INSTALL_SH.format(**ctx), encoding="utf-8")
    (root / "install.sh").chmod(0o755)
    (root / "docker-compose.yml").write_text(COMPOSE.format(**ctx), encoding="utf-8")
    (root / "README.md").write_text(README.format(**ctx), encoding="utf-8")

    zip_path = Path(out_dir) / f"{name}-dmr.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            zf.write(p, p.relative_to(root.parent))
    return zip_path
