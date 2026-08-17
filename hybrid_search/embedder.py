"""文本嵌入器。

所有嵌入器统一返回稠密向量 `list[float]`。默认 HashEmbedder 零依赖；
SentenceTransformerEmbedder 需要 `sentence-transformers` 和 `torch`，懒加载。
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter

from .tokenize import tokenize


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


class HashEmbedder:
    """字符 n-gram 哈希稀疏向量（默认，零依赖、可复现）。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for term, count in Counter(tokenize(text)).items():
            idx = _hash_token(term, self.dim)
            vec[idx] += 1.0 + math.log(count)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """真正的语义向量，需要 `sentence-transformers`。

    embedder = SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        # 国内网络直连 huggingface.co 通常不可达；默认改走镜像，用户已显式设置时尊重其配置。
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from sentence_transformers import SentenceTransformer  # 懒加载

        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


def make_embedder(prefer_semantic: bool = True):
    """默认优先返回真正的语义嵌入器；未安装 sentence-transformers 时回退哈希嵌入器。"""
    if prefer_semantic:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    return HashEmbedder()
