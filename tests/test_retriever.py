import unittest

from hybrid_search.retriever import Document, HybridRetriever


DOCS = [
    Document(id="a", title="向量语义检索", text="把文本编码成向量，用余弦相似度找语义相近内容。"),
    Document(id="b", title="BM25 关键词检索", text="BM25 按词频和逆文档频率计算相关性，精确匹配关键词。"),
    Document(id="c", title="RRF 混合", text="RRF 把 BM25 和向量结果按排名倒数融合。"),
]


class RetrieverTest(unittest.TestCase):
    def test_retrieve_returns_documents(self):
        retriever = HybridRetriever(DOCS)
        hits = retriever.retrieve("向量和 BM25 怎么融合", top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertIn("c", [d.id for d in hits])

    def test_bm25_prefers_exact_term(self):
        retriever = HybridRetriever(DOCS)
        scores = retriever._bm25_scores(["bm25"])
        best = max(range(len(DOCS)), key=lambda i: scores[i])
        self.assertEqual(DOCS[best].id, "b")


if __name__ == "__main__":
    unittest.main()
