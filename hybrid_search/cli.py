"""命令行入口：python -m hybrid_search.cli "查询词" [--top-k N] [--rag]"""

from __future__ import annotations

import argparse

from .rag import RAGGenerator
from .retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--rag", action="store_true", help="用 LLM 生成带来源引用的答案")
    args = parser.parse_args()

    retriever = HybridRetriever.from_jsonl(args.corpus)
    docs = retriever.retrieve(args.query, top_k=args.top_k)
    for i, doc in enumerate(docs, start=1):
        print(f"{i}. [{doc.id}] {doc.title}")
        print(f"   {doc.text[:80]}")

    if args.rag:
        result = RAGGenerator().generate(args.query, docs)
        print("\n--- 回答 ---")
        print(result["answer"])


if __name__ == "__main__":
    main()
