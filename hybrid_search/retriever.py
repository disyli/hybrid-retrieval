"""混合检索器：BM25 + 稀疏向量，RRF 融合，可选标题加权。"""

from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .embedder import HashEmbedder
from .tokenize import tokenize


@dataclass
class Document:
    id: str
    title: str
    text: str


class HybridRetriever:
    def __init__(self, documents: list[Document], embedder=None, rrf_k: int = 60) -> None:
        self.documents = documents
        self.embedder = embedder or HashEmbedder()
        self.rrf_k = rrf_k
        self.tokens = [tokenize(d.title + " " + d.text) for d in documents]
        self.bm25 = BM25(self.tokens)
        self.vecs = [self.embedder.embed(toks) for toks in self.tokens]

    def _bm25_scores(self, qtok: list[str]) -> list[float]:
        return [self.bm25.score(qtok, i) for i in range(len(self.documents))]

    def _vector_scores(self, qtok: list[str]) -> list[float]:
        qvec = self.embedder.embed(qtok)
        return [self.embedder.cosine(qvec, v) for v in self.vecs]

    @staticmethod
    def _rrf_rank_scores(scores: list[float]) -> list[float]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked = [0.0] * len(scores)
        for rank, idx in enumerate(order):
            if scores[idx] > 0:
                ranked[idx] = 1.0 / (rank + 1)
        return ranked

    def retrieve(self, query: str, top_k: int = 5, rerank: bool = True) -> list[Document]:
        qtok = tokenize(query)
        bm25 = self._bm25_scores(qtok)
        vec = self._vector_scores(qtok)

        rrf_bm25 = self._rrf_rank_scores(bm25)
        rrf_vec = self._rrf_rank_scores(vec)
        fused = [self.rrf_k + a + b for a, b in zip(rrf_bm25, rrf_vec)]

        if rerank:
            qset = set(qtok)
            for i, doc in enumerate(self.documents):
                title_tokens = set(tokenize(doc.title))
                if qset & title_tokens:
                    fused[i] += 0.1  # 标题命中轻量加权

        order = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
        return [self.documents[i] for i in order[:top_k]]

    @staticmethod
    def from_jsonl(path: str) -> "HybridRetriever":
        import json

        docs: list[Document] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                docs.append(Document(id=obj["id"], title=obj["title"], text=obj["text"]))
        return HybridRetriever(docs)
