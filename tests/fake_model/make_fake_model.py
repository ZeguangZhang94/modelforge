"""端到端测试辅助：生成一个最小合法的 GGUF 假模型文件。"""

import struct
from pathlib import Path


def wstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def make_fake_gguf(path: Path, quant_name: str = "q4_k_m") -> None:
    """构造一个只有元数据头 + 少量填充数据的最小 GGUF 文件。"""
    kvs = [
        ("general.architecture", 8, wstr("llama")),          # string
        ("general.name", 8, wstr("tiny-llama-e2e")),
        ("general.license", 8, wstr("mit")),
        ("general.size_label", 8, wstr("0.1B")),
        ("llama.block_count", 4, struct.pack("<I", 8)),       # uint32
        ("llama.embedding_length", 4, struct.pack("<I", 512)),
        ("llama.context_length", 4, struct.pack("<I", 4096)),
    ]
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))          # version
        f.write(struct.pack("<Q", 0))          # tensor_count
        f.write(struct.pack("<Q", len(kvs)))   # kv_count
        for key, vtype, val in kvs:
            f.write(wstr(key))
            f.write(struct.pack("<I", vtype))
            f.write(val)
        # 填充 ~1MB 假权重数据，让文件有体积
        f.write(b"\x00" * (1 << 20))


if __name__ == "__main__":
    out = Path(__file__).parent
    make_fake_gguf(out / "tiny-llama-q4_k_m.gguf")
    make_fake_gguf(out / "tiny-llama-q8_0.gguf", "q8_0")
    (out / "config.json").write_text(
        '{"model_type": "llama", "max_position_embeddings": 4096}', encoding="utf-8")
    (out / "LICENSE").write_text("MIT License (fake model for e2e test)\n", encoding="utf-8")
    print(f"假模型已生成于 {out}")
