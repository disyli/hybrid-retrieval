"""标准 Okapi BM25 实现，纯标准库。"""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(1, self.n_docs)
        self.doc_freqs = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for freq in self.doc_freqs:
            df.update(freq.keys())
        self.idf = {
            term: math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0)
            for term, n in df.items()
        }

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        qfreq = Counter(query_tokens)
        dl = self.doc_len[doc_idx]
        df = self.doc_freqs[doc_idx]
        score = 0.0
        for term, qtf in qfreq.items():
            if term not in df:
                continue
            tf = df[term]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * tf * (self.k1 + 1) / denom * qtf
        return score
