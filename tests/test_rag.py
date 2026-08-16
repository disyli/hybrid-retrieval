import unittest

from hybrid_search.rag import RAGGenerator
from hybrid_search.retriever import Document


class RAGGeneratorTest(unittest.TestCase):
    def test_falls_back_without_api_key(self):
        gen = RAGGenerator(api_key="", base_url="https://example.com", model="test")
        result = gen.generate("查询", [Document(id="1", title="t", text="内容")])
        self.assertEqual(result["status"], "retrieval_only")
        self.assertIn("内容", result["answer"])


if __name__ == "__main__":
    unittest.main()
