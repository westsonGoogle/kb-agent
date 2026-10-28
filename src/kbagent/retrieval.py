"""BM25 检索:不用向量库,靠字/词混合分词把中文检索做得够用。

为什么不直接按空格切词?中文没有空格。这里用「连续汉字取 bigram」
(知识库 -> 知识/识库),既保留词组信息,又不依赖 jieba 之类的词典。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .store import ChunkRecord

_LATIN = re.compile(r"[a-z0-9_]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")

# 疑问词/功能词对检索没有信息量,只在查询侧过滤,不影响索引
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "not",
    "this",
    "that",
    "what",
    "how",
    "why",
    "which",
    "who",
    "when",
    "where",
    "it",
    "its",
    "as",
    "at",
    "by",
    "from",
    "can",
    "could",
    "should",
    "would",
    "will",
    "do",
    "does",
    "did",
    "about",
    "into",
    "怎么",
    "怎样",
    "如何",
    "什么",
    "哪些",
    "哪个",
    "为什么",
    "是否",
    "能否",
    "可以",
    "能够",
    "这个",
    "那个",
    "这些",
    "那些",
    "我们",
    "你们",
    "他们",
    "它们",
    "以及",
    "还是",
    "或者",
    "但是",
    "因为",
    "所以",
    "如果",
    "已经",
    "需要",
    "进行",
    "使用",
    "通过",
    "一个",
    "一些",
    "没有",
    "就是",
    "不是",
    "不能",
    "可能",
    "应该",
    "的话",
    "请问",
    "告诉",
    "一下",
    "相关",
    "内容",
    "问题",
    "文档",
    "资料",
    "具体",
    "主要",
    "一般",
    "时候",
}


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = _LATIN.findall(lowered)
    for run in _CJK.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def query_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


@dataclass(slots=True)
class Hit:
    chunk_id: int
    doc_id: int
    source: str
    title: str
    text: str
    start_line: int
    score: float
    page: int | None = None

    @property
    def location(self) -> str:
        if self.page:
            return f"第 {self.page} 页"
        return f"第 {self.start_line} 行"


class BM25Index:
    def __init__(self, chunks: list[ChunkRecord], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._freqs = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self._lengths = [sum(freq.values()) for freq in self._freqs]
        self._titles = [set(tokenize(chunk.title)) for chunk in chunks]

        total = sum(self._lengths)
        self._avgdl = total / len(chunks) if chunks else 0.0

        document_freq: Counter[str] = Counter()
        for freq in self._freqs:
            document_freq.update(freq.keys())
        count = len(chunks)
        self._idf = {
            term: math.log(1 + (count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = 6, *, max_per_source: int = 2) -> list[Hit]:
        terms = query_tokens(query)
        if not terms or not self.chunks:
            return []

        query_freq = Counter(terms)
        term_set = set(terms)
        results: list[Hit] = []

        for index, chunk in enumerate(self.chunks):
            freq = self._freqs[index]
            length = self._lengths[index]
            score = 0.0
            for term, query_count in query_freq.items():
                tf = freq.get(term)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * length / self._avgdl)
                score += self._idf[term] * tf * (self.k1 + 1) / denominator * query_count
            if score <= 0:
                continue
            # 标题命中给一点加权:问「缓存」时缓存专题文档更容易排前面
            if self._titles[index] & term_set:
                score *= 1.15
            results.append(
                Hit(
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    source=chunk.source,
                    title=chunk.title,
                    text=chunk.text,
                    start_line=chunk.start_line,
                    page=chunk.page,
                    score=score,
                )
            )

        results.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        if not results:
            return []
        # 丢掉明显不相关的结果:低于最高分 30% 的片段对回答只有噪声
        cutoff = results[0].score * 0.3
        results = [hit for hit in results if hit.score >= cutoff]

        # 同一份文档最多两段,避免一份 README 把 top-k 全占了
        selected: list[Hit] = []
        per_source: dict[str, int] = {}
        for hit in results:
            if per_source.get(hit.source, 0) >= max_per_source:
                continue
            per_source[hit.source] = per_source.get(hit.source, 0) + 1
            selected.append(hit)
            if len(selected) >= max(1, top_k):
                break
        return selected

    def document_tokens(self, query: str) -> list[str]:
        """调试用:看看查询被切成了什么。"""
        return tokenize(query)
