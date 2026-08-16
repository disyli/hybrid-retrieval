"""在标注查询集上评估 BM25 / 向量 / 混合三种召回，输出 recall@k 和 MRR。"""

from __future__ import annotations

import json

from hybrid_search.retriever import HybridRetriever
from hybrid_search.tokenize import tokenize


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at_k(hits: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for h in hits[:k] if h in relevant) / len(relevant)


def mrr(hits: list[str], relevant: list[str]) -> float:
    for rank, h in enumerate(hits, start=1):
        if h in relevant:
            return 1.0 / rank
    return 0.0


def main() -> None:
    retriever = HybridRetriever.from_jsonl("data/corpus.jsonl")
    queries = load_jsonl("data/queries.jsonl")

    k = 5
    rows = []
    for mode in ("bm25", "vector", "hybrid"):
        total_recall = 0.0
        total_mrr = 0.0
        for q in queries:
            if mode == "bm25":
                scores = retriever._bm25_scores(q["query"])
            elif mode == "vector":
                scores = retriever._vector_scores(q["query"])
            else:
                docs = retriever.retrieve(q["query"], top_k=k)
                hits = [d.id for d in docs]
                total_recall += recall_at_k(hits, q["relevant"], k)
                total_mrr += mrr(hits, q["relevant"])
                continue
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            hits = [retriever.documents[i].id for i in order[:k]]
            total_recall += recall_at_k(hits, q["relevant"], k)
            total_mrr += mrr(hits, q["relevant"])
        rows.append((mode, total_recall / len(queries), total_mrr / len(queries)))

    print(f"{'mode':<8} {'recall@{k}':>10} {'MRR':>8}")
    for mode, recall, mrr_score in rows:
        print(f"{mode:<8} {recall:>10.3f} {mrr_score:>8.3f}")


if __name__ == "__main__":
    main()
