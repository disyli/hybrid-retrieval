"""中文友好的分词：对中文按字符 unigram + bigram 切分，对英文按字母数字串切分。"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """返回带重复的词元列表（供 BM25 统计词频）。"""
    text = text.lower()
    tokens: list[str] = []
    pos = 0
    for match in _CJK.finditer(text):
        if match.start() > pos:
            tokens.extend(_ALNUM.findall(text[pos : match.start()]))
        run = match.group()
        tokens.extend(run)  # 单字
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # 双字
        pos = match.end()
    if pos < len(text):
        tokens.extend(_ALNUM.findall(text[pos:]))
    return tokens
