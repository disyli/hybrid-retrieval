"""向量近邻索引。默认 FlatIndex 纯 Python 暴力检索；FaissIndex 懒加载 faiss。"""

from __future__ import annotations

from .embedder import cosine


class FlatIndex:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        scored = [(i, cosine(query, v)) for i, v in enumerate(self.vectors)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


class FaissIndex:
    """基于 faiss 的 IndexFlatIP（内积 = 余弦，前提是向量已归一化）。

    需要 `faiss-cpu` 和 `numpy`；懒加载，未安装时构造会抛 ImportError。
    """

    def __init__(self, vectors: list[list[float]]) -> None:
        import faiss  # 懒加载
        import numpy as np

        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        import faiss  # 懒加载
        import numpy as np

        q = np.asarray([query], dtype="float32")
        faiss.normalize_L2(q)
        distances, indices = self._index.search(q, k)
        return list(zip(indices[0].tolist(), distances[0].tolist()))


def make_index(vectors: list[list[float]], backend: str = "auto"):
    if backend == "faiss":
        try:
            return FaissIndex(vectors)
        except ImportError:
            return FlatIndex(vectors)
    if backend == "flat":
        return FlatIndex(vectors)
    try:
        return FaissIndex(vectors)
    except ImportError:
        return FlatIndex(vectors)
