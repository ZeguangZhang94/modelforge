#!/usr/bin/env python3
"""极简本地 RAG：纯标准库 TF-IDF 检索，零依赖、完全离线。

支持 txt / md / docx（zip+XML 提取）文档：
- 分块（500 字符，50 重叠）
- TF-IDF 向量化 + 余弦相似度检索
- 文档存 state/docs/，索引随用随建（小文档集足够快）
"""

from __future__ import annotations

import math
import re
import zipfile
from collections import Counter
from pathlib import Path

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[一-鿿]", re.UNICODE)


def extract_text(path: Path) -> str:
    """提取文档纯文本。docx 走 zip+XML，其余按文本读。"""
    if path.suffix.lower() == ".docx":
        try:
            with zipfile.ZipFile(path) as zf:
                xml = zf.read("word/document.xml").decode("utf-8", "replace")
            text = re.sub(r"<w:p[ >]", "\n", xml)
            return re.sub(r"<[^>]+>", "", text)
        except Exception:
            return ""
    return path.read_text(encoding="utf-8", errors="replace")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def chunk_text(text: str) -> list[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


class TfidfIndex:
    def __init__(self) -> None:
        self.chunks: list[tuple[str, str]] = []  # (文档名, 块文本)
        self.tfs: list[Counter] = []
        self.df: Counter = Counter()

    def add_document(self, name: str, text: str) -> int:
        n = 0
        for c in chunk_text(text):
            tf = Counter(tokenize(c))
            if not tf:
                continue
            self.chunks.append((name, c))
            self.tfs.append(tf)
            for tok in tf:
                self.df[tok] += 1
            n += 1
        return n

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not self.chunks:
            return []
        n_docs = len(self.chunks)
        q_tf = Counter(tokenize(query))
        scores = []
        for i, tf in enumerate(self.tfs):
            score = 0.0
            for tok, qf in q_tf.items():
                if tok not in tf:
                    continue
                idf = math.log((n_docs + 1) / (self.df[tok] + 1)) + 1
                score += qf * tf[tok] * idf * idf
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [{"doc": self.chunks[i][0], "text": self.chunks[i][1],
                 "score": round(s, 2)} for s, i in scores[:k]]


def build_index(docs_dir: Path) -> TfidfIndex:
    idx = TfidfIndex()
    if docs_dir.exists():
        for p in sorted(docs_dir.iterdir()):
            if p.suffix.lower() in (".txt", ".md", ".docx"):
                text = extract_text(p)
                if text.strip():
                    idx.add_document(p.name, text)
    return idx
