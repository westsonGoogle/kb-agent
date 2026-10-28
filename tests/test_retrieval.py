from __future__ import annotations

import pytest

from kbagent.retrieval import BM25Index, tokenize
from kbagent.store import ChunkRecord


def make_chunks() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            id=1,
            doc_id=1,
            source="caching.md",
            title="caching.md",
            text="prompt cache 要求前缀完全一致。时间戳会破坏缓存命中。",
            start_line=10,
        ),
        ChunkRecord(
            id=2,
            doc_id=2,
            source="fastapi.md",
            title="fastapi.md",
            text="FastAPI 用 StreamingResponse 返回 text/event-stream 实现流式输出。",
            start_line=20,
        ),
        ChunkRecord(
            id=3,
            doc_id=3,
            source="incident.md",
            title="incident.md",
            text="线上 500 排查顺序:时间线、错误分布、日志、资源、回滚。",
            start_line=3,
        ),
    ]


def test_tokenize_mixes_chinese_bigrams_and_latin_words() -> None:
    tokens = tokenize("Prompt Caching 命中率")
    assert "prompt" in tokens
    assert "caching" in tokens
    assert "命中" in tokens
    assert "中率" in tokens


def test_search_chinese() -> None:
    index = BM25Index(make_chunks())
    hits = index.search("缓存 命中")
    assert hits
    assert hits[0].source == "caching.md"
    assert hits[0].location == "第 10 行"


def test_search_english() -> None:
    index = BM25Index(make_chunks())
    hits = index.search("StreamingResponse 流式")
    assert hits[0].source == "fastapi.md"


def test_search_top_k_and_empty_query() -> None:
    index = BM25Index(make_chunks())
    assert len(index.search("排查 日志", top_k=1)) == 1
    assert index.search("") == []
    assert index.search("量子 纠缠 半导体") == []


def test_page_location() -> None:
    chunks = [
        ChunkRecord(
            id=1,
            doc_id=1,
            source="manual.pdf",
            title="manual.pdf",
            text="数据库连接耗尽时先统计连接来源。",
            start_line=1,
            page=7,
        )
    ]
    hit = BM25Index(chunks).search("数据库连接", top_k=1)[0]
    assert hit.location == "第 7 页"


def test_title_boost_does_not_break_ranking() -> None:
    index = BM25Index(make_chunks())
    hits = index.search("fastapi")
    assert hits[0].source == "fastapi.md"


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_returns_empty(query: str) -> None:
    assert BM25Index(make_chunks()).search(query) == []


def test_stopword_only_query_returns_empty() -> None:
    assert BM25Index(make_chunks()).search("怎么 如何 哪些") == []


def test_relative_cutoff_drops_noise() -> None:
    hits = BM25Index(make_chunks()).search("排查 日志", top_k=10)
    assert len(hits) == 1
    assert hits[0].source == "incident.md"


def test_max_two_chunks_per_source() -> None:
    chunks = [
        ChunkRecord(
            id=index,
            doc_id=1,
            source="same.md",
            title="same.md",
            text="缓存 命中 " * 10,
            start_line=index,
        )
        for index in range(1, 6)
    ]
    hits = BM25Index(chunks).search("缓存", top_k=6)
    assert len(hits) == 2
    assert all(hit.source == "same.md" for hit in hits)
