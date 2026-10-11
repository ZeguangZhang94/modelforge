"""ModelPack 规范（modelpack/1）——框架与安装器之间的唯一契约。

纯标准库实现：dataclasses + 手写校验，保证框架与安装器零依赖。
对应文档：docs/02-架构设计.md 第 2.5 节。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "modelpack/1"

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(-[a-z0-9.]+)?$")

ARCHITECTURES = ("dense", "moe", "mamba", "hybrid")
BACKENDS = ("llama-cpp", "strata-moe", "vllm")
SOURCE_TYPES = ("local", "huggingface", "modelscope", "url")
CAPABILITIES = ("chat", "vision", "thinking", "tool_use", "embedding")


class ModelPackError(ValueError):
    """ModelPack 校验或解析失败。"""


@dataclass
class Publisher:
    name: str = ""
    contact: str = ""


@dataclass
class ModelInfo:
    family: str
    architecture: str  # dense | moe | mamba | hybrid
    license: str = "unknown"
    capabilities: list[str] = field(default_factory=lambda: ["chat"])
    layers: int = 0           # 总层数（引擎参数计算用）
    expert_count: int = 0     # MoE 专家总数（0 = 稠密）
    expert_used_count: int = 0  # 单 token 激活专家数


@dataclass
class Requirements:
    min_ram_gb: float
    vram_hot_gb: float
    disk_gb: float
    recommended_ram_gb: float = 0.0  # 0 = 与 min 相同


@dataclass
class VariantSource:
    type: str  # local | huggingface | modelscope | url
    # local:   path 指向打包时复制进安装包 models/ 的相对路径（M1 支持）
    # hf/ms:   repo + files 通配；url: 直链列表
    path: str = ""
    repo: str = ""
    files: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    sha256: dict[str, str] = field(default_factory=dict)  # 文件名 → 摘要


@dataclass
class Variant:
    id: str
    size_gb: float
    quality_rank: int  # 1 最糊，越大越好
    requirements: Requirements
    sources: list[VariantSource] = field(default_factory=list)
    format: str = "gguf"  # gguf | safetensors | awq | gptq


@dataclass
class EngineSpec:
    backend: str  # llama-cpp | strata-moe | vllm
    version_pin: dict[str, str] = field(default_factory=dict)  # {"ref": ..., "sha256": ...}
    defaults: dict[str, Any] = field(default_factory=lambda: {
        "context": 8192,
        "speculative_decode": False,
        "parallel": 1,
    })


@dataclass
class ServiceSpec:
    port: int = 8080
    apis: list[str] = field(default_factory=lambda: ["openai", "anthropic", "health", "metrics"])
    lan_exposure: str = "opt-in-with-key"


@dataclass
class Branding:
    """安装包品牌自定义：名称、主题色、欢迎语、许可证文案、关于信息。"""
    display_name: str = ""        # 界面显示名（默认用包名）
    logo_text: str = "⚡"          # 标题旁 Logo（emoji 或短文本）
    theme_color: str = "#4c9aff"  # 主题色（hex）
    welcome_message: str = ""     # 控制台首条欢迎语
    about: str = ""               # 关于/说明文字（控制台与向导展示）
    license_text: str = ""        # 自定义许可证文案（安装向导首屏展示）


@dataclass
class ModelPack:
    name: str
    version: str
    model: ModelInfo
    variants: list[Variant]
    engine: EngineSpec
    publisher: Publisher = field(default_factory=Publisher)
    service: ServiceSpec = field(default_factory=ServiceSpec)
    branding: Branding = field(default_factory=Branding)
    checksums_sha256: dict[str, str] = field(default_factory=dict)
    signature: Optional[str] = None
    schema: str = SCHEMA_VERSION

    # ---------- 序列化 ----------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # schema 字段放最前，便于人读
        return {"schema": d.pop("schema"), **d}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent) + "\n"

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    # ---------- 反序列化 ----------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelPack":
        try:
            pack = cls(
                name=d["name"],
                version=d["version"],
                model=ModelInfo(**d["model"]),
                variants=[
                    Variant(
                        id=v["id"],
                        size_gb=v["size_gb"],
                        quality_rank=v["quality_rank"],
                        requirements=Requirements(**v["requirements"]),
                        sources=[VariantSource(**s) for s in v.get("sources", [])],
                        format=v.get("format", "gguf"),
                    )
                    for v in d["variants"]
                ],
                engine=EngineSpec(**d["engine"]),
                publisher=Publisher(**d.get("publisher", {})),
                service=ServiceSpec(**d.get("service", {})),
                branding=Branding(**d.get("branding", {})),
                checksums_sha256=d.get("checksums_sha256", {}),
                signature=d.get("signature"),
                schema=d.get("schema", SCHEMA_VERSION),
            )
        except (KeyError, TypeError) as e:
            raise ModelPackError(f"modelpack 字段缺失或类型错误: {e}") from e
        pack.validate()
        return pack

    @classmethod
    def load(cls, path: str | Path) -> "ModelPack":
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ModelPackError(f"{p}: 不是合法 JSON: {e}") from e
        return cls.from_dict(data)

    # ---------- 校验 ----------

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema != SCHEMA_VERSION:
            errors.append(f"schema 必须是 {SCHEMA_VERSION}，实际为 {self.schema}")
        if not _NAME_RE.match(self.name):
            errors.append(f"name 非法: {self.name!r}（小写字母/数字开头，可含 . _ -，2~64 字符）")
        if not _SEMVER_RE.match(self.version):
            errors.append(f"version 非法: {self.version!r}（需为 semver，如 1.0.0）")
        if self.model.architecture not in ARCHITECTURES:
            errors.append(f"architecture 非法: {self.model.architecture}")
        bad_caps = set(self.model.capabilities) - set(CAPABILITIES)
        if bad_caps:
            errors.append(f"未知 capabilities: {sorted(bad_caps)}")
        if self.engine.backend not in BACKENDS:
            errors.append(f"engine.backend 非法: {self.engine.backend}")
        if not self.variants:
            errors.append("variants 不能为空")
        ids = [v.id for v in self.variants]
        if len(ids) != len(set(ids)):
            errors.append("variants 存在重复 id")
        for v in self.variants:
            if v.size_gb <= 0:
                errors.append(f"variant {v.id}: size_gb 必须 > 0")
            if v.requirements.vram_hot_gb < 0 or v.requirements.min_ram_gb <= 0:
                errors.append(f"variant {v.id}: requirements 数值非法")
            if not v.sources:
                errors.append(f"variant {v.id}: 至少需要一个下载/文件来源")
            for s in v.sources:
                if s.type not in SOURCE_TYPES:
                    errors.append(f"variant {v.id}: source.type 非法: {s.type}")
        if not (1 <= self.service.port <= 65535):
            errors.append(f"service.port 非法: {self.service.port}")
        if self.branding.theme_color and not re.match(
                r"^#[0-9a-fA-F]{3}([0-9a-fA-F]{3})?$", self.branding.theme_color):
            errors.append(f"branding.theme_color 需为 hex 颜色: {self.branding.theme_color}")
        if errors:
            raise ModelPackError("ModelPack 校验失败:\n  - " + "\n  - ".join(errors))
