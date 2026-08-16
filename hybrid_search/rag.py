"""带来源引用的 RAG 生成。

LLM 通过 OpenAI 兼容的 /chat/completions 接入；未配置 API Key 时降级为
“仅检索”模式，直接返回片段拼接，不调用模型。

环境变量：LLM_API_KEY（或 OPENAI_API_KEY）、LLM_BASE_URL、LLM_MODEL。
"""

from __future__ import annotations

import json
import os
import urllib.request

from .models import Document


class RAGGenerator:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"

    def generate(self, query: str, docs: list[Document]) -> dict:
        sources = [{"id": d.id, "title": d.title, "snippet": d.text[:200]} for d in docs]
        if not self.api_key:
            answer = "\n\n".join(f"[{i}] {d.title}\n{d.text}" for i, d in enumerate(docs, start=1))
            return {"status": "retrieval_only", "answer": answer, "sources": sources}

        passages = "\n\n".join(
            f"[{i}] {d.title}\n{d.text}" for i, d in enumerate(docs, start=1)
        )
        user_prompt = (
            "根据下面的资料回答用户问题。只能引用资料内容，并在句末标注来源编号 [1]、[2]。\n"
            f"用户问题：{query}\n\n资料：\n{passages}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是严谨的知识库问答助手，回答必须有据可查。"},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        answer = data["choices"][0]["message"]["content"]
        return {"status": "ok", "answer": answer, "sources": sources}
