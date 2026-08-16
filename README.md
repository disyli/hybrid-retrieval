# hybrid-retrieval

一个极简、零依赖的混合检索演示：**BM25 词法检索 + 稀疏向量检索 + RRF 倒排融合 + 轻量标题重排**，附带标注查询集上的 recall@k / MRR 评估脚本。

它的目的是把一个"最小可用、可评估、可替换"的检索链路讲清楚——从数据接入、两种召回、结果融合到评估，而不是调一个黑盒检索接口。

## 为什么默认用哈希嵌入而不是真模型

为了让仓库 clone 下来就能跑、不依赖 GPU 和大模型下载，默认嵌入器 `HashEmbedder` 用**字符 n-gram 哈希 + 子线性词频 + L2 归一化**生成确定性稀疏向量。它偏词法，能体现"向量化检索"的结构，但不是真正的语义向量。

要换成真正的语义模型，只需在 `hybrid_search/embedder.py` 里新增一个实现同接口的嵌入器，例如：

```python
from sentence_transformers import SentenceTransformer

class STEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()
```

然后把 `HybridRetriever(embedder=STEmbedder())` 传进去即可，其余代码不变。

## 目录结构

```text
hybrid_search/
  tokenize.py     中文按字符 unigram+bigram 切分，英文按字母数字串
  bm25.py         Okapi BM25
  embedder.py     默认哈希嵌入器 + 真模型接入说明
  retriever.py    BM25/向量/RRF 融合/标题重排
  cli.py          命令行检索入口
evaluate.py       recall@k 与 MRR 评估
data/             示例语料和标注查询
tests/            最小冒烟测试
```

## 快速开始

环境要求只有 Python 3.9+，无第三方依赖。

```bash
# 命令行检索
python -m hybrid_search.cli "BM25 和向量怎么合并" --top-k 5

# 评估三种召回
python evaluate.py

# 跑测试
python -m unittest discover -s tests
```

## 检索链路

1. 查询和文档统一分词（中文单字 + 双字，英文单词）。
2. 两条独立召回：BM25 走词频逆文档频率，向量走 n-gram 哈希嵌入 + 余弦相似度。
3. 对两条结果各自做 RRF 排名分数（`1 / (k + rank)`），相加融合，避开两种分数量纲不一致的问题。
4. 轻量重排：查询词命中标题时加一个固定权重。
5. 返回 top-k 文档，评估脚本按标注相关集统计 recall@k 和 MRR。

## 下一步可以扩展的方向

- 把 `HashEmbedder` 换成 sentence-transformers，得到真正的语义召回。
- 加 FAISS / HNSW 做近邻索引，支持更大规模。
- 加交叉编码器重排，替换当前的规则重排。
- 把召回结果接进一个 LLM，做带来源引用的 RAG 生成。
