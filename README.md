# hybrid-retrieval

一个极简的混合检索与 RAG 演示：**BM25 词法检索 + 向量检索 + RRF 倒排融合 + 重排 + 带来源引用的 LLM 生成**，附带标注查询集上的 recall@k / MRR 评估脚本。

它的目的是把一个"最小可用、可评估、可替换"的检索链路讲清楚——从数据接入、两种召回、结果融合到评估，而不是调一个黑盒检索接口。

## 为什么保留哈希嵌入作为兜底

为了让仓库 clone 下来就能跑、不依赖 GPU 和大模型下载，未安装依赖时的兜底嵌入器 `HashEmbedder` 用**字符 n-gram 哈希 + 子线性词频 + L2 归一化**生成确定性稀疏向量。它偏词法，能体现"向量化检索"的结构，但不是真正的语义向量。

安装 `sentence-transformers` 后无需改代码：默认嵌入器自动升级为 `SentenceTransformerEmbedder`（真语义向量），默认重排器自动升级为 `CrossEncoderReranker`（交叉编码器精排）；见 `make_embedder()` / `make_reranker()`。

## 目录结构

```text
hybrid_search/
  tokenize.py     中文按字符 unigram+bigram 切分，英文按字母数字串
  bm25.py         Okapi BM25
  embedder.py     哈希嵌入器 + sentence-transformers 语义嵌入器
  vector_index.py FlatIndex（暴力）/ FaissIndex（近邻索引）
  reranker.py     规则重排 / 交叉编码器重排
  retriever.py    BM25/向量/RRF 融合
  rag.py          OpenAI 兼容 LLM 生成，带来源引用
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

# 带 LLM 生成（需要配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL）
python -m hybrid_search.cli "BM25 和向量怎么合并" --top-k 5 --rag

# 评估三种召回
python evaluate.py

# 跑测试
python -m unittest discover -s tests
```

## 检索链路

1. 查询和文档统一分词（中文单字 + 双字，英文单词）。
2. 两条独立召回：BM25 走词频逆文档频率，向量走语义嵌入 + 余弦相似度（兜底为 n-gram 哈希嵌入）。
3. 对两条结果各自做 RRF 排名分数（`1 / (k + rank)`），相加融合，避开两种分数量纲不一致的问题。
4. 重排：默认交叉编码器对候选做联合精排（未装依赖时回退为"查询词命中标题加权"的规则重排）。
5. 返回 top-k 文档，评估脚本按标注相关集统计 recall@k 和 MRR。

## 四项可插拔能力

默认仓库零第三方依赖、clone 即可跑；下面四项都是"装了依赖/配了 key 就启用，否则走兜底"，其中语义向量与交叉编码器重排在安装后即为默认。

### 1. 真正的语义向量

```bash
pip install sentence-transformers torch
```

```python
from hybrid_search.embedder import SentenceTransformerEmbedder
from hybrid_search.retriever import HybridRetriever

# 安装后 HybridRetriever(docs) 默认即用语义向量；显式传入可换模型
retriever = HybridRetriever(docs, embedder=SentenceTransformerEmbedder())
```

### 2. FAISS 近邻索引

```bash
pip install faiss-cpu numpy
```

```python
retriever = HybridRetriever(docs, index_backend="faiss")
```

`index_backend="auto"` 时优先 faiss，没装则回退纯 Python 暴力检索。

### 3. 交叉编码器重排

```bash
pip install sentence-transformers torch
```

```python
from hybrid_search.reranker import CrossEncoderReranker

# 安装后 HybridRetriever(docs) 默认即用交叉编码器重排；显式传入可换模型
retriever = HybridRetriever(docs, reranker=CrossEncoderReranker())
```

### 4. 带来源引用的 RAG 生成

通过 OpenAI 兼容的 `/chat/completions` 接入，配置环境变量即可：

```bash
export LLM_API_KEY="..."
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
```

```python
from hybrid_search.rag import RAGGenerator

docs = retriever.retrieve("BM25 和向量怎么合并", top_k=5)
result = RAGGenerator().generate("BM25 和向量怎么合并", docs)
print(result["answer"])  # 带 [1][2] 来源标注
```

未配置 `LLM_API_KEY` 时，`generate` 降级为 `retrieval_only`，直接返回片段拼接，不调用模型。
