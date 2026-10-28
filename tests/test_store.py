from __future__ import annotations

from pathlib import Path

import pytest

from kbagent.store import KBStore


def make_store(tmp_path: Path, *, chunk_chars: int = 300) -> tuple[KBStore, Path]:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    store = KBStore(tmp_path / "kb.db", docs_dir, chunk_chars=chunk_chars, chunk_overlap=50)
    return store, docs_dir


def test_ingest_and_read(tmp_path: Path) -> None:
    store, docs = make_store(tmp_path)
    path = docs / "a.md"
    path.write_text("第一段内容。\n\n第二段内容。\n\n第三段内容。", encoding="utf-8")

    info = store.ingest(path)
    assert info.source == "a.md"
    assert info.chunk_count >= 1
    assert store.count_chunks() == info.chunk_count

    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0].title == "a.md"

    chunks = store.load_chunks()
    assert chunks[0].source == "a.md"
    assert "第一段" in chunks[0].text

    full = store.document_text("a.md")
    assert "第三段" in str(full["content"])


def test_reingest_replaces_chunks(tmp_path: Path) -> None:
    store, docs = make_store(tmp_path)
    path = docs / "a.md"
    path.write_text("旧内容", encoding="utf-8")
    first = store.ingest(path)

    path.write_text("新内容,更长一些。" * 20, encoding="utf-8")
    second = store.ingest(path)

    assert first.id == second.id
    assert len(store.list_documents()) == 1
    assert all("旧内容" not in chunk.text for chunk in store.load_chunks())


def test_nested_source_path(tmp_path: Path) -> None:
    store, docs = make_store(tmp_path)
    nested = docs / "sub"
    nested.mkdir()
    path = nested / "b.md"
    path.write_text("嵌套目录里的内容", encoding="utf-8")

    info = store.ingest(path)
    assert info.source == "sub/b.md"


def test_delete_removes_file_and_rows(tmp_path: Path) -> None:
    store, docs = make_store(tmp_path)
    path = docs / "a.md"
    path.write_text("待删除内容", encoding="utf-8")
    info = store.ingest(path)

    assert store.delete_document(info.id) is True
    assert not path.exists()
    assert store.list_documents() == []
    assert store.delete_document(info.id) is False


def test_reindex_adds_and_removes(tmp_path: Path) -> None:
    store, docs = make_store(tmp_path)
    (docs / "a.md").write_text("文档 A", encoding="utf-8")
    (docs / "sub").mkdir()
    (docs / "sub" / "b.md").write_text("文档 B", encoding="utf-8")
    (docs / "ignore.csv").write_text("a,b", encoding="utf-8")

    first = store.reindex()
    assert first.indexed == 2
    assert first.chunks >= 2

    (docs / "a.md").unlink()
    second = store.reindex()
    assert second.removed == 1
    assert [doc.source for doc in store.list_documents()] == ["sub/b.md"]


def test_unsupported_suffix(tmp_path: Path) -> None:
    store, docs = make_store(tmp_path)
    path = docs / "a.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持"):
        store.ingest(path)


def test_document_text_by_title_and_ambiguous(tmp_path: Path) -> None:
    store, docs = make_store(tmp_path)
    (docs / "sub1").mkdir()
    (docs / "sub1" / "a.md").write_text("A1 的内容", encoding="utf-8")
    (docs / "sub2").mkdir()
    (docs / "sub2" / "a.md").write_text("A2 的内容", encoding="utf-8")
    store.reindex()

    assert "A1 的内容" in str(store.document_text("sub1/a.md")["content"])
    with pytest.raises(ValueError, match="匹配到多份"):
        store.document_text("a.md")
