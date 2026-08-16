"""命令行入口：python -m hybrid_search.cli "查询词" [--top-k N] [--corpus path]"""

from __future__ import annotations

import argparse

from .retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    args = parser.parse_args()

    retriever = HybridRetriever.from_jsonl(args.corpus)
    for i, doc in enumerate(retriever.retrieve(args.query, top_k=args.top_k), start=1):
        print(f"{i}. [{doc.id}] {doc.title}")
        print(f"   {doc.text[:80]}")


if __name__ == "__main__":
    main()
