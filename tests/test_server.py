from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentlab.config import Settings
from agentlab.llm import ToolCall
from fakes import FakeLLM, text_script, tool_script
from fastapi.testclient import TestClient

from kbagent.agent import build_agent
from kbagent.config import KBSettings
from kbagent.kb import KnowledgeBase
from kbagent.server import create_app

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
    knowledge.reindex()
    return knowledge


def make_client(kb: KnowledgeBase) -> TestClient:
    def factory(base: KnowledgeBase, citations):
        agent = build_agent(base, citations, SETTINGS)
        agent.llm = FakeLLM(
            [
                tool_script(
                    ToolCall(
                        id="call_1",
                        name="search_kb",
                        arguments='{"query": "缓存 命中"}',
                    )
                ),
                text_script("缓存要求前缀一致 [1]。"),
            ]
        )
        return agent

    return TestClient(create_app(kb=kb, settings=SETTINGS, agent_factory=factory))


def parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if line.startswith("data:")]
        if not lines:
            continue
        events.append(json.loads(lines[0][len("data:") :].strip()))
    return events


def test_index_and_health(kb: KnowledgeBase) -> None:
    client = make_client(kb)
    page = client.get("/")
    assert page.status_code == 200
    assert "kb-agent" in page.text

    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["documents"] == 1
    assert health["chunks"] >= 1


def test_chat_streams_tool_call_and_sources(kb: KnowledgeBase) -> None:
    client = make_client(kb)
    with client.stream("POST", "/api/chat", json={"message": "缓存怎么命中?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = parse_sse(body)
    types = [event["type"] for event in events]
    assert "session" in types
    assert "tool_call" in types
    assert "tool_result" in types
    assert "done" in types
    assert "sources" in types
    assert types[-1] == "end"

    tool_call = next(event for event in events if event["type"] == "tool_call")
    assert tool_call["name"] == "search_kb"
    assert tool_call["arguments"]["query"] == "缓存 命中"

    sources = next(event for event in events if event["type"] == "sources")["sources"]
    assert sources[0]["cite"] == 1
    assert sources[0]["source"] == "caching.md"

    done = next(event for event in events if event["type"] == "done")
    assert "[1]" in done["text"]


def test_upload_and_delete(kb: KnowledgeBase) -> None:
    client = make_client(kb)
    response = client.post(
        "/api/upload",
        files={"file": ("uploaded.md", "这是上传的新文档。".encode(), "text/markdown")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "uploaded.md"
    assert payload["chunk_count"] >= 1

    documents = client.get("/api/documents").json()["documents"]
    assert len(documents) == 2
    uploaded = next(item for item in documents if item["source"] == "uploaded.md")

    assert client.delete(f"/api/documents/{uploaded['id']}").json() == {"deleted": True}
    assert len(client.get("/api/documents").json()["documents"]) == 1
    assert not (kb.settings.docs_dir / "uploaded.md").exists()


def test_upload_rejects_unsupported_type(kb: KnowledgeBase) -> None:
    client = make_client(kb)
    response = client.post(
        "/api/upload",
        files={"file": ("data.csv", b"a,b", "text/csv")},
    )
    assert response.status_code == 400


def test_reindex_endpoint(kb: KnowledgeBase) -> None:
    client = make_client(kb)
    response = client.post("/api/reindex")
    assert response.status_code == 200
    payload = response.json()
    assert payload["indexed"] == 1
    assert payload["chunks"] >= 1


def test_empty_message_rejected(kb: KnowledgeBase) -> None:
    client = make_client(kb)
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
