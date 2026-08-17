"""重排器。默认 make_reranker() 优先交叉编码器（需 sentence-transformers），否则回退规则重排。"""

from __future__ import annotations

import os

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
    """交叉编码器重排，需要 `sentence-transformers`，懒加载。

    默认 BAAI/bge-reranker-base：多语言（中文友好）的交叉编码器重排模型。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        # 国内网络直连 huggingface.co 通常不可达；默认改走镜像，用户已显式设置时尊重其配置。
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from sentence_transformers import CrossEncoder  # 懒加载

        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: list[Document], scores: list[float]) -> list[Document]:
        pairs = [(query, f"{d.title}\n{d.text[:400]}") for d in candidates]
        ce_scores = self._model.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: ce_scores[i], reverse=True)
        return [candidates[i] for i in order]

def make_reranker(prefer_cross_encoder: bool = True) -> Reranker:
    """默认优先交叉编码器重排；未安装 sentence-transformers（或模型不可用）时回退规则重排。"""
    if prefer_cross_encoder:
        try:
            return CrossEncoderReranker()
        except Exception:
            pass
    return RuleReranker()
