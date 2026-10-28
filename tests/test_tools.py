from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentlab.config import Settings

from kbagent.agent import build_agent
from kbagent.citations import CitationRegistry
from kbagent.config import KBSettings
from kbagent.kb import KnowledgeBase

SETTINGS = Settings(api_key="test", base_url="http://llm.test/v1", model="fake-model")


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    settings = KBSettings(
        data_dir=tmp_path,
        docs_dir=tmp_path / "docs",
        db_path=tmp_path / "kb.db",
        chunk_chars=400,
        chunk_overlap=60,
    )
    knowledge = KnowledgeBase(settings)
    (settings.docs_dir / "caching.md").write_text(
        "prompt cache 要求前缀完全一致。\n时间戳会破坏缓存命中。", encoding="utf-8"
    )
    (settings.docs_dir / "fastapi.md").write_text(
        "FastAPI 用 StreamingResponse 返回 text/event-stream 实现流式输出。",
        encoding="utf-8",
    )
    knowledge.reindex()
    return knowledge


def make_tools(kb: KnowledgeBase) -> tuple[object, CitationRegistry]:
    citations = CitationRegistry()
    agent = build_agent(kb, citations, SETTINGS)
    return agent.tools, citations


def test_search_tool_returns_citations_and_dedupes(kb: KnowledgeBase) -> None:
    tools, citations = make_tools(kb)
    first = json.loads(tools.execute("search_kb", '{"query": "缓存 命中"}'))
    assert first["count"] >= 1
    assert first["results"][0]["cite"] == 1
    assert first["results"][0]["source"] == "caching.md"
    assert len(citations) == first["count"]

    second = json.loads(tools.execute("search_kb", '{"query": "缓存 命中"}'))
    assert second["results"][0]["cite"] == 1
    assert len(citations) == first["count"]


def test_search_tool_assigns_new_cite_for_new_query(kb: KnowledgeBase) -> None:
    tools, citations = make_tools(kb)
    tools.execute("search_kb", '{"query": "缓存"}')
    count_before = len(citations)

    result = json.loads(tools.execute("search_kb", '{"query": "StreamingResponse"}'))
    assert result["results"][0]["cite"] == count_before + 1


def test_search_tool_miss_returns_hint(kb: KnowledgeBase) -> None:
    tools, _ = make_tools(kb)
    result = json.loads(tools.execute("search_kb", '{"query": "量子计算"}'))
    assert result["results"] == []
    assert "hint" in result


def test_open_document_tool(kb: KnowledgeBase) -> None:
    tools, _ = make_tools(kb)
    result = json.loads(tools.execute("open_document", '{"source": "caching.md"}'))
    assert result["source"] == "caching.md"
    assert "前缀" in result["content"]
    assert result["total_lines"] == 2


def test_open_document_unknown(kb: KnowledgeBase) -> None:
    tools, _ = make_tools(kb)
    result = tools.execute("open_document", '{"source": "missing.md"}')
    assert "找不到文档" in result


def test_list_documents_tool(kb: KnowledgeBase) -> None:
    tools, _ = make_tools(kb)
    result = json.loads(tools.execute("list_documents", "{}"))
    assert result["total"] == 2
    sources = {item["source"] for item in result["documents"]}
    assert sources == {"caching.md", "fastapi.md"}


def test_tool_schemas_registered(kb: KnowledgeBase) -> None:
    tools, _ = make_tools(kb)
    names = {schema["function"]["name"] for schema in tools.schemas()}
    assert names == {"search_kb", "open_document", "list_documents"}
