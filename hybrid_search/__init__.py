"""极简混合检索演示：BM25 + 稀疏向量 + RRF 融合 + 轻量重排。

零第三方依赖，默认嵌入器是确定性的字符 n-gram 哈希稀疏向量；
如需真正的语义向量，把 embedder 换成 sentence-transformers 即可（见 embedder.py）。
"""

__version__ = "0.1.0"
