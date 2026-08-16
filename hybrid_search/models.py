"""共享数据模型，避免 retriever / reranker / rag 之间的循环导入。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    id: str
    title: str
    text: str
