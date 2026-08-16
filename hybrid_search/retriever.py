"""混合检索器：BM25 + 向量检索，RRF 融合，重排器排序。"""

from __future__ import annotations

from .bm25 import BM25
from .embedder import HashEmbedder
from .models import Document
from .reranker import Reranker, RuleReranker
from .tokenize import tokenize
from .vector_index import make_index


class HybridRetriever:
    def __init__(
        self,
        documents: list[Document],
        embedder=None,
        index_backend: str = "auto",
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        self.documents = documents
        self.embedder = embedder or HashEmbedder()
        self.reranker = reranker or RuleReranker()
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.tokens = [tokenize(d.title + " " + d.text) for d in documents]
        self.bm25 = BM25(self.tokens)
        vectors = [self.embedder.embed(d.title + " " + d.text) for d in documents]
        self.index = make_index(vectors, backend=index_backend)

    def _bm25_scores(self, query_text: str) -> list[float]:
        qtok = tokenize(query_text)
        return [self.bm25.score(qtok, i) for i in range(len(self.documents))]

    def _vector_scores(self, query_text: str) -> list[float]:
        qvec = self.embedder.embed(query_text)
        scores = [0.0] * len(self.documents)
        for idx, score in self.index.search(qvec, len(self.documents)):
            scores[idx] = score
        return scores

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        n = len(self.documents)
        k = min(self.candidate_k, n)

        bm25 = self._bm25_scores(query)
        bm25_order = sorted(range(n), key=lambda i: bm25[i], reverse=True)[:k]

        vec_hits = self.index.search(self.embedder.embed(query), k)

        rrf: dict[int, float] = {}
        for rank, idx in enumerate(bm25_order, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (idx, _) in enumerate(vec_hits, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        cand_idx = sorted(rrf, key=lambda i: rrf[i], reverse=True)
        candidates = [self.documents[i] for i in cand_idx]
        scores = [rrf[i] for i in cand_idx]
        ranked = self.reranker.rerank(query, candidates, scores)
        return ranked[:top_k]

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
