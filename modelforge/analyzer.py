"""模型分析器：从模型目录识别架构、参数量、量化变体、能力标签。

支持两类输入：
- HuggingFace 风格目录（含 config.json，权重为 safetensors/bin）
- GGUF 目录（一个或多个 .gguf 文件，直接读取 GGUF 元数据头）

对应文档：docs/02-架构设计.md 第 2.3 节。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelProfile:
    family: str
    architecture: str                 # dense | moe | mamba | hybrid
    params_b: float                   # 总参数量（十亿）
    active_params_b: float            # 单 token 激活参数（稠密 = 总参数）
    context_max: int
    quant_variants: list[dict]        # [{"id": "q4_k_m", "file": "...", "size_gb": ...}]
    capabilities: list[str] = field(default_factory=lambda: ["chat"])
    engine_backend: str = "llama-cpp"
    license: str = "unknown"
    source_config: dict = field(default_factory=dict)


# ---------------- GGUF 元数据读取（最小实现） ----------------

_GGUF_VALUE_READERS = {}


def _read_gguf_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype: int):
    # GGUF 类型：0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool,8 str,9 arr,10 u64,11 i64,12 f64
    fmt_map = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
               6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if vtype == 8:
        return _read_gguf_string(f)
    if vtype == 9:  # array：只支持字符串数组，其余跳过
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if elem_type == 8:
            return [_read_gguf_string(f) for _ in range(count)]
        if elem_type in fmt_map:
            size = struct.calcsize(fmt_map[elem_type])
            f.read(size * count)
            return f"<{count} items>"
        raise ValueError(f"不支持的 GGUF 数组元素类型 {elem_type}")
    if vtype in fmt_map:
        return struct.unpack(fmt_map[vtype], f.read(struct.calcsize(fmt_map[vtype])))[0]
    raise ValueError(f"不支持的 GGUF 元数据类型 {vtype}")


def read_gguf_metadata(path: Path) -> dict:
    """读取 GGUF 文件的元数据键值对（不读张量数据）。"""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path.name} 不是 GGUF 文件")
        (version,) = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise ValueError(f"不支持的 GGUF 版本 {version}")
        f.read(8)  # tensor_count
        (kv_count,) = struct.unpack("<Q", f.read(8))
        meta = {}
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            meta[key] = _read_gguf_value(f, vtype)
        return meta


def read_gguf_tensor_flags(path: Path) -> dict[str, bool]:
    """扫描 GGUF 张量名，检测内嵌能力（MTP 草稿层等）。不读张量数据。

    GGUF 布局：头部 + KV 元数据之后是张量信息表：
    name(string), n_dims(u32), dims(u64×n), type(u32), offset(u64)
    """
    flags = {"has_mtp": False}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return flags
            (version,) = struct.unpack("<I", f.read(4))
            if version not in (2, 3):
                return flags
            (tensor_count,) = struct.unpack("<Q", f.read(8))
            (kv_count,) = struct.unpack("<Q", f.read(8))
            # 跳过 KV 元数据（复用现有读取器的跳过逻辑）
            fmt_map = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                       10: 8, 11: 8, 12: 8}
            for _ in range(kv_count):
                (n,) = struct.unpack("<Q", f.read(8))
                f.read(n)
                (vt,) = struct.unpack("<I", f.read(4))
                if vt == 8:
                    (m,) = struct.unpack("<Q", f.read(8))
                    f.read(m)
                elif vt == 9:
                    (et,) = struct.unpack("<I", f.read(4))
                    (cnt,) = struct.unpack("<Q", f.read(8))
                    if et == 8:
                        for _ in range(cnt):
                            (m,) = struct.unpack("<Q", f.read(8))
                            f.read(m)
                    elif et in fmt_map:
                        f.read(fmt_map[et] * cnt)
                    else:
                        return flags
                elif vt in fmt_map:
                    f.read(fmt_map[vt])
                else:
                    return flags
            # 扫描张量名
            for _ in range(min(tensor_count, 20000)):
                (n,) = struct.unpack("<Q", f.read(8))
                name = f.read(n).decode("utf-8", "replace").lower()
                (n_dims,) = struct.unpack("<I", f.read(4))
                f.read(8 * n_dims + 4 + 8)  # dims + type + offset
                if "nextn" in name or ".mtp" in name or "mtp." in name:
                    flags["has_mtp"] = True
                    break
    except Exception:
        pass
    return flags


# ---------------- 量化档位识别 ----------------

_KNOWN_QUANTS = [
    # (文件名关键词, 档位 id, 位宽近似值, 质量排序)
    ("q2_0", "q2_0", 2.6, 2), ("iq2_xs", "iq2_xs", 2.5, 3),
    ("iq3_xxs", "iq3_xxs", 3.1, 4), ("iq3_s", "iq3_s", 3.4, 5),
    ("q3_k_m", "q3_k_m", 3.9, 5), ("q4_0", "q4_0", 4.6, 5),
    ("q4_k_m", "q4_k_m", 4.9, 6), ("iq4_xs", "iq4_xs", 4.5, 6),
    ("q5_k_m", "q5_k_m", 5.7, 7), ("q6_k", "q6_k", 6.6, 8),
    ("q8_0", "q8_0", 8.5, 9), ("f16", "f16", 16.0, 10), ("bf16", "bf16", 16.0, 10),
]


def _guess_quant(filename: str) -> tuple[str, float, int] | None:
    low = filename.lower()
    for kw, qid, bits, rank in _KNOWN_QUANTS:
        if kw in low:
            return qid, bits, rank
    return None


# ---------------- 主入口 ----------------

def analyze(model_dir: str | Path) -> ModelProfile:
    """分析本地模型目录，产出 ModelProfile。"""
    d = Path(model_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {d}")

    ggufs = sorted(d.glob("*.gguf"))
    config_path = d / "config.json"

    if ggufs:
        return _analyze_gguf_dir(d, ggufs, config_path)
    if config_path.exists():
        return _analyze_hf_dir(d, config_path)
    raise ValueError(f"{d} 既不是 GGUF 目录也没有 config.json，无法识别模型")


def _analyze_gguf_dir(d: Path, ggufs: list[Path], config_path: Path) -> ModelProfile:
    meta = read_gguf_metadata(ggufs[0])
    tensor_flags = read_gguf_tensor_flags(ggufs[0])
    arch = str(meta.get("general.architecture", "unknown"))
    family = str(meta.get("general.name", d.name))
    license_ = str(meta.get("general.license", "unknown"))

    n_expert = int(meta.get(f"{arch}.expert_count", 0) or 0)
    n_expert_used = int(meta.get(f"{arch}.expert_used_count", 0) or 0)
    n_layer = int(meta.get(f"{arch}.block_count", 0) or 0)
    n_embd = int(meta.get(f"{arch}.embedding_length", 0) or 0)
    ctx = int(meta.get(f"{arch}.context_length", 8192) or 8192)

    # 参数量估算：优先元数据，否则按文件大小粗估
    params_b = 0.0
    size_label = str(meta.get("general.size_label", ""))
    if size_label:
        import re
        m = re.search(r"([\d.]+)\s*([BM])", size_label, re.I)
        if m:
            params_b = float(m.group(1)) * (1.0 if m.group(2).upper() == "B" else 0.001)
    if params_b == 0.0:
        # 粗估：用最大量化文件大小 × 2（按 ~4bit 反推 fp16 参数）
        biggest = max(ggufs, key=lambda p: p.stat().st_size)
        params_b = biggest.stat().st_size * 2 / 1e9

    is_moe = n_expert > 0
    active_b = params_b * (n_expert_used / n_expert) if is_moe and n_expert_used else params_b

    # 分片 GGUF 归组 + 草稿/投影文件分离
    # gpt-oss-120b-Q2_K_L-00001-of-00002.gguf → 一组两文件，引擎只给首片
    import re as _re
    shard_re = _re.compile(r"^(.+?)-(\d{5})-of-(\d{5})\.gguf$", _re.I)
    shard_groups: dict[str, list[Path]] = {}
    main_ggufs: list[Path] = []
    draft_file = ""
    for g in ggufs:
        low = g.name.lower()
        if "mmproj" in low:
            continue  # 多模态投影：随包但不作为档位
        if "eagle3" in low or "draft" in low:
            draft_file = g.name  # 投机解码草稿模型
            continue
        m = shard_re.match(g.name)
        if m:
            shard_groups.setdefault(m.group(1), []).append(g)
        else:
            main_ggufs.append(g)

    variants = []
    for g in main_ggufs:
        q = _guess_quant(g.name)
        variants.append({
            "id": q[0] if q else g.stem,
            "file": g.name,
            "files": [g.name],
            "size_gb": round(g.stat().st_size / 1e9, 4),
            "bits": q[1] if q else 4.9,
            "quality_rank": q[2] if q else 5,
        })
    for base, shards in sorted(shard_groups.items()):
        shards = sorted(shards, key=lambda p: p.name)
        total = sum(p.stat().st_size for p in shards)
        q = _guess_quant(shards[0].name)
        variants.append({
            "id": q[0] if q else base,
            "file": shards[0].name,           # 引擎入口：首片（llama.cpp 自动找同组）
            "files": [p.name for p in shards],
            "size_gb": round(total / 1e9, 4),
            "bits": q[1] if q else 4.9,
            "quality_rank": q[2] if q else 5,
        })

    if params_b == 0.0:
        # 粗估：用最大变体总大小 × 2（按 ~4bit 反推 fp16 参数）
        biggest_gb = max((v["size_gb"] for v in variants), default=0.0)
        params_b = biggest_gb * 2

    caps = ["chat"]
    if (d / "mmproj.gguf").exists() or any("mmproj" in g.name.lower() for g in ggufs):
        caps.append("vision")

    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            ctx = int(cfg.get("max_position_embeddings", ctx))
        except Exception:
            pass

    return ModelProfile(
        family=family,
        architecture="moe" if is_moe else "dense",
        params_b=round(params_b, 1),
        active_params_b=round(active_b, 1),
        context_max=ctx,
        quant_variants=variants,
        capabilities=caps,
        engine_backend="strata-moe" if is_moe else "llama-cpp",
        license=license_,
        source_config={
            "arch": arch, "layers": n_layer, "embd": n_embd,
            "expert_count": n_expert, "expert_used": n_expert_used,
            "has_mtp": tensor_flags["has_mtp"],
            "draft_file": draft_file,
        },
    )


def _analyze_hf_dir(d: Path, config_path: Path) -> ModelProfile:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(cfg.get("model_type", "unknown"))
    n_expert = int(cfg.get("num_experts", cfg.get("n_routed_experts", 0)) or 0)
    n_expert_used = int(cfg.get("num_experts_per_tok", cfg.get("num_experts_per_token", 0)) or 0)
    ctx = int(cfg.get("max_position_embeddings", 8192))

    total_bytes = sum(p.stat().st_size for p in d.glob("*.safetensors"))
    total_bytes += sum(p.stat().st_size for p in d.glob("*.bin"))
    # fp16/bf16 权重 → 参数量 ≈ 字节数 / 2
    params_b = total_bytes / 2 / 1e9 if total_bytes else 0.0
    is_moe = n_expert > 0
    active_b = params_b * (n_expert_used / n_expert) if is_moe and n_expert_used else params_b

    license_ = "unknown"
    for cand in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        if (d / cand).exists():
            license_ = "见随附 LICENSE 文件"
            break

    return ModelProfile(
        family=d.name,
        architecture="moe" if is_moe else "dense",
        params_b=round(params_b, 1),
        active_params_b=round(active_b, 1),
        context_max=ctx,
        quant_variants=[{
            "id": "fp16", "file": "*.safetensors",
            "size_gb": round(total_bytes / 1e9, 2),
            "bits": 16.0, "quality_rank": 10,
        }] if total_bytes else [],
        capabilities=["chat"],
        engine_backend="strata-moe" if is_moe else "llama-cpp",
        license=license_,
        source_config={
            "model_type": model_type,
            "expert_count": n_expert, "expert_used": n_expert_used,
        },
    )
