from __future__ import annotations

from kbagent.citations import CitationRegistry
from kbagent.retrieval import Hit


def make_hit(text: str, chunk_id: int = 1) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=1,
        source="a.md",
        title="a.md",
        text=text,
        start_line=1,
        score=1.0,
    )


def test_snippet_is_centered_on_query() -> None:
    text = "无关内容。" * 200 + "prompt caching 的命中率。" + "尾巴。" * 100
    registry = CitationRegistry()
    registry.add(make_hit(text), "prompt caching")

    snippet = registry.items()[0].snippet
    assert "prompt caching" in snippet
    assert len(snippet) <= 242
    assert snippet.startswith("…")


def test_dedupe_and_numbering() -> None:
    registry = CitationRegistry()
    first = registry.add(make_hit("内容 A", 1), "A")
    again = registry.add(make_hit("内容 A", 1), "A")
    second = registry.add(make_hit("内容 B", 2), "B")

    assert first == 1
    assert again == 1
    assert second == 2
    assert len(registry) == 2
    assert [item["cite"] for item in registry.as_dicts()] == [1, 2]


def test_clear() -> None:
    registry = CitationRegistry()
    registry.add(make_hit("内容", 1), "内容")
    registry.clear()
    assert len(registry) == 0
    assert registry.add(make_hit("内容", 1), "内容") == 1
