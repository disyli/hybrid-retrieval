"""极简混合检索演示：BM25 + 向量 + RRF 融合 + 重排。

零第三方依赖即可跑（兜底为哈希嵌入 + 规则重排）；安装 sentence-transformers 后，
默认自动升级为语义向量 + 交叉编码器重排（见 embedder.py / reranker.py）。
"""

__version__ = "0.1.0"
