"""打包器：把模型目录 + 安装框架运行时骨架组装成可分发的安装包。

M1 产物形态：便携 zip（跨平台），内含：
    <name>-<version>/
    ├── modelpack.json        # ModelPack 清单（唯一契约）
    ├── installer.py          # 通用安装器（与模型无关，清单驱动）
    ├── START-HERE.bat        # Windows 双击入口
    ├── install.sh            # Linux/macOS 单命令入口
    ├── models/               # 本地来源的权重文件（远程来源则不含）
    └── licenses/             # 许可证合集

对应文档：docs/02-架构设计.md 第 2.7 节。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from .analyzer import ModelProfile, analyze
from .estimator import estimate_all
from .modelpack import (Branding, EngineSpec, ModelInfo, ModelPack, Publisher,
                        Variant, VariantSource)

RUNTIME_DIR = Path(__file__).parent / "runtime_bundle"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_pack(
    model_dir: str | Path,
    name: str,
    version: str = "1.0.0",
    publisher: str = "",
    context: int = 8192,
    engine_tag: str = "",
    branding: Branding | None = None,
) -> tuple[ModelPack, ModelProfile]:
    """分析模型目录并生成 ModelPack（不落盘）。"""
    model_dir = Path(model_dir)
    profile = analyze(model_dir)
    estimates = estimate_all(profile, context)
    if not estimates:
        raise ValueError("未找到任何可用的量化变体（目录里没有 .gguf 或 safetensors 权重）")

    variants: list[Variant] = []
    for v, est in zip(profile.quant_variants, estimates):
        src = VariantSource(type="local", path=f"models/{v['file']}",
                            files=v.get("files") or [v["file"]])
        variants.append(Variant(
            id=v["id"],
            size_gb=v["size_gb"],
            quality_rank=v.get("quality_rank", 5),
            requirements=est.requirements,
            sources=[src],
            format="gguf" if v["file"].endswith(".gguf") else "safetensors",
        ))

    if all(v.format == "safetensors" for v in variants):
        raise ValueError(
            "该模型是 safetensors 格式，推理引擎无法直接运行。\n"
            "请先转换为 GGUF：python -m modelforge convert <模型目录>\n"
            "（或在打包台「模型商店」直接下载现成的 GGUF 版本）")

    pack = ModelPack(
        name=name,
        version=version,
        model=ModelInfo(
            family=profile.family,
            architecture=profile.architecture,
            license=profile.license,
            capabilities=profile.capabilities,
            layers=int(profile.source_config.get("layers", 0)),
            expert_count=int(profile.source_config.get("expert_count", 0)),
            expert_used_count=int(profile.source_config.get("expert_used", 0)),
        ),
        variants=variants,
        engine=EngineSpec(
            backend=profile.engine_backend,
            version_pin={"ref": engine_tag or ""},
            defaults={
                "context": min(context, profile.context_max),
                "speculative_decode": True,
                # 草稿优先级：独立草稿文件（eagle3 等）> 内嵌 MTP 层 > ngram-simple
                "spec_type": ("draft-eagle3" if "eagle3" in
                              profile.source_config.get("draft_file", "").lower()
                              else "draft-mtp" if profile.source_config.get("has_mtp")
                              else "ngram-simple"),
                "draft_model": profile.source_config.get("draft_file", ""),
                "parallel": 1,
            },
        ),
        publisher=Publisher(name=publisher),
        branding=branding or Branding(),
    )
    pack.validate()
    return pack, profile


def assemble(
    pack: ModelPack,
    model_dir: str | Path,
    out_dir: str | Path,
) -> Path:
    """组装安装包目录，返回包目录路径。"""
    model_dir = Path(model_dir)
    root = Path(out_dir) / f"{pack.name}-{pack.version}"
    if root.exists():
        shutil.rmtree(root)
    (root / "models").mkdir(parents=True)
    (root / "licenses").mkdir()

    # 1. 复制模型权重（local 来源，支持分片变体）并计算校验
    checksums: dict[str, str] = {}
    for variant in pack.variants:
        for src in variant.sources:
            if src.type != "local":
                continue
            fnames = src.files or [Path(src.path).name]
            for fname in fnames:
                src_file = model_dir / fname
                if not src_file.exists():
                    raise FileNotFoundError(f"变体 {variant.id} 的权重文件缺失: {src_file}")
                dst = root / "models" / fname
                if not dst.exists():
                    shutil.copy2(src_file, dst)
                digest = sha256_file(dst)
                checksums[f"models/{fname}"] = digest
                src.sha256[fname] = digest

    # mmproj（多模态投影）与草稿模型（eagle3/mtp）随包
    for extra in list(model_dir.glob("mmproj*.gguf")) + list(model_dir.glob("*eagle3*.gguf")) + list(model_dir.glob("*draft*.gguf")):
        dst = root / "models" / extra.name
        if not dst.exists():
            shutil.copy2(extra, dst)
        checksums[f"models/{extra.name}"] = sha256_file(dst)

    # 许可证
    for cand in model_dir.glob("LICENSE*"):
        shutil.copy2(cand, root / "licenses" / cand.name)

    # 2. 清单（先写一次拿摘要字段，再定稿）
    pack.checksums_sha256 = checksums
    pack.save(root / "modelpack.json")

    # 3. 运行时骨架：通用安装器 + 引擎管理器 + 资源规划器 + 服务注册 + 卸载辅助 + API 网关 + 控制台 + 平台入口
    for f in ("installer.py", "engine_manager.py", "planner.py", "service.py",
              "rag.py", "uninstall_helper.py", "mcp_server.py",
              "serve.py", "console.html", "START-HERE.bat", "install.sh"):
        shutil.copy2(RUNTIME_DIR / f, root / f)
    write_deploy_kit(root, pack)
    (root / "README.txt").write_text(
        f"{pack.name} {pack.version}\n"
        f"{'=' * 40}\n\n"
        f"模型: {pack.model.family}（{pack.model.architecture}）\n"
        f"许可证: {pack.model.license}\n\n"
        f"安装：\n"
        f"  Windows: 双击 START-HERE.bat\n"
        f"  Linux/macOS: ./install.sh\n\n"
        f"安装完成后打开 http://127.0.0.1:{pack.service.port}\n"
        f"API（OpenAI 兼容）: http://127.0.0.1:{pack.service.port}/v1\n",
        encoding="utf-8",
    )
    return root


def bundle_engines(root: Path, engine_dir: str | Path) -> int:
    """把本地引擎二进制目录打进安装包 engines/（离线可用）。返回文件数。"""
    src = Path(engine_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"引擎目录不存在: {src}")
    dst = root / "engines"
    dst.mkdir(exist_ok=True)
    n = 0
    for p in src.iterdir():
        if p.is_file() and p.suffix in (".exe", ".dll", "", ".so"):
            if p.suffix == ".part" or p.name.endswith(".zip"):
                continue
            import shutil as _sh
            _sh.copy2(p, dst / p.name)
            n += 1
    return n


def bundle_python(root: Path, embed_zip: str | Path) -> None:
    """把 Python embeddable 运行时打进安装包 runtime/python/（终端零依赖，Windows）。

    Embeddable 版是官方免安装 Python，我们的运行时纯标准库，无需 pip。
    """
    src = Path(embed_zip)
    if not src.exists():
        raise FileNotFoundError(f"Python embeddable 不存在: {src}")
    dst = root / "runtime" / "python"
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dst)


DEPLOY_DETECTION_PS1 = r"""# ModelForge 安装检测脚本（Intune Win32 检测规则 / SCCM 用）
# 约定：已安装 → exit 0；未安装 → exit 1
$dir = "$env:LOCALAPPDATA\{name}"
if (Test-Path "$dir\state\config.json") {{ exit 0 }} else {{ exit 1 }}
"""

DEPLOYMENT_MD = """# 企业批量部署指南（{name}）

## 支持矩阵

| 平台 | 方式 | 命令 |
|---|---|---|
| SCCM / MECM | EXE 脚本部署（无包大小限制） | `START-HERE.bat /S /DIR=D:\\AI\\{name}` |
| Intune | Win32 包装（≤8GB 包适用） | 同上，检测规则用 `deploy/detection.ps1` |
| GPO 启动脚本 | 计算机启动脚本 | 同上 |
| 手动/试点 | 双击 | `START-HERE.bat`（浏览器向导） |

## 静默安装契约

```
START-HERE.bat /S [/DIR=<安装目录>] [/NOSTART]
```

| 返回码 | 含义 |
|---|---|
| 0 | 成功 |
| 1603 | 安装失败（硬件不满足/校验失败等，日志在安装目录 state/logs/） |

- `/S`：完全静默，全部接受推荐值（自动选卡、推荐档位）
- `/DIR=`：自定义安装目录（默认 `%LOCALAPPDATA%\\{name}`）
- `/NOSTART`：装完不注册登录自启
- 幂等：重复执行不会重复安装

## 卸载

```
START-HERE.bat --uninstall --yes
```

## 注意

- 本包大于 8GB 时不要用 Intune Win32（上限 8GB），走 SCCM 或文件共享+启动脚本
- 安装全程零出站网络请求（权重与引擎已内置），适合隔离网
- 完整校验清单见 modelpack.json 的 checksums_sha256 字段
"""


def write_deploy_kit(root: Path, pack: ModelPack) -> None:
    """企业部署套件：检测脚本 + 部署指南（所有包标配）。"""
    d = root / "deploy"
    d.mkdir(exist_ok=True)
    (d / "detection.ps1").write_text(
        DEPLOY_DETECTION_PS1.format(name=pack.name), encoding="utf-8")
    (d / "DEPLOYMENT.md").write_text(
        DEPLOYMENT_MD.format(name=pack.name), encoding="utf-8")


def build_update_pack(old_dir: str | Path, new_dir: str | Path,
                      out_dir: str | Path) -> Path:
    """离线增量更新包：对比两个包目录的运行时文件差异，产出更新 zip。

    只含变化的运行时/清单文件（不含模型权重），带 SHA256 清单。
    摆渡流程：联网机器生成 → 校验 → 介质拷贝 → 离线机 --apply-update。
    """
    old_dir, new_dir = Path(old_dir), Path(new_dir)
    old_mp = json.loads((old_dir / "modelpack.json").read_text(encoding="utf-8"))
    new_mp = json.loads((new_dir / "modelpack.json").read_text(encoding="utf-8"))

    runtime_files = ["installer.py", "engine_manager.py", "planner.py",
                     "service.py", "rag.py", "uninstall_helper.py",
                     "mcp_server.py", "serve.py", "console.html",
                     "START-HERE.bat", "install.sh", "modelpack.json"]
    changed = []
    for rel in runtime_files:
        nf = new_dir / rel
        if not nf.exists():
            continue
        of = old_dir / rel
        digest = sha256_file(nf)
        if not of.exists() or sha256_file(of) != digest:
            changed.append({"path": rel, "sha256": digest})
    if not changed:
        raise ValueError("两个版本运行时无差异，无需更新包")

    manifest = {"type": "modelforge-update", "name": new_mp["name"],
                "from_version": old_mp["version"], "to_version": new_mp["version"],
                "files": changed}
    zip_path = Path(out_dir) / f"{new_mp['name']}-update-{old_mp['version']}-to-{new_mp['version']}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for f in changed:
            zf.write(new_dir / f["path"], f["path"])
    return zip_path


def zip_package(root: Path) -> Path:
    """把包目录打成 zip，返回 zip 路径。

    模型权重/引擎压缩包用 STORED（基本不可压缩，DEFLATED 纯属浪费时间），
    其余小文件用 DEFLATED。
    """
    zip_path = root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as zf:
        for p in sorted(root.rglob("*")):
            comp = zipfile.ZIP_STORED if p.suffix.lower() in (
                ".gguf", ".zip", ".tar.gz") else zipfile.ZIP_DEFLATED
            zf.write(p, p.relative_to(root.parent), compress_type=comp)
    return zip_path
