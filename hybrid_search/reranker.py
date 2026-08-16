"""重排器。默认 RuleReranker（规则），可选 CrossEncoderReranker（交叉编码器）。"""

from __future__ import annotations

from .models import Document
from .tokenize import tokenize


class Reranker:
    def rerank(self, query: str, candidates: list[Document], scores: list[float]) -> list[Document]:
        raise NotImplementedError


class RuleReranker(Reranker):
    """规则重排：在融合分数之上，查询词命中标题时加固定权重。"""

    def rerank(self, query: str, candidates: list[Document], scores: list[float]) -> list[Document]:
        qtok = set(tokenize(query))
        combined = [
            s + (0.1 if qtok & set(tokenize(d.title)) else 0.0)
            for s, d in zip(scores, candidates)
        ]
        order = sorted(range(len(candidates)), key=lambda i: combined[i], reverse=True)
        return [candidates[i] for i in order]


class CrossEncoderReranker(Reranker):
    """交叉编码器重排，需要 `sentence-transformers`，懒加载。"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # 懒加载

        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: list[Document], scores: list[float]) -> list[Document]:
        pairs = [(query, f"{d.title}\n{d.text[:400]}") for d in candidates]
        ce_scores = self._model.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: ce_scores[i], reverse=True)
        return [candidates[i] for i in order]
