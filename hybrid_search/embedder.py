"""嵌入器。

默认 HashEmbedder：把字符 n-gram 哈希到固定维度、用子线性词频加权并 L2 归一化，
得到确定性的稀疏向量。它偏词法、不是语义向量，但零依赖、可复现。

要换真正的语义模型，安装 sentence-transformers 后实现如下接口即可：

    class STEmbedder:
        def __init__(self):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        def embed(self, text: str) -> list[float]:
            return self.model.encode(text).tolist()
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


class HashEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, tokens: list[str]) -> dict[int, float]:
        vec: dict[int, float] = {}
        for term, count in Counter(tokens).items():
            idx = _hash_token(term, self.dim)
            weight = 1.0 + math.log(count)  # 子线性词频
            vec[idx] = vec.get(idx, 0.0) + weight
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {idx: v / norm for idx, v in vec.items()}

    @staticmethod
    def cosine(a: dict[int, float], b: dict[int, float]) -> float:
        if not a or not b:
            return 0.0
        small, large = (a, b) if len(a) <= len(b) else (b, a)
        return sum(v * large.get(k, 0.0) for k, v in small.items())
