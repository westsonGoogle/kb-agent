"""知识库服务:存储 + 索引 + 检索的统一入口,CLI / Web / 工具共用。"""

from __future__ import annotations

import threading
from pathlib import Path

from .config import KBSettings
from .retrieval import BM25Index, Hit
from .store import DocumentInfo, IndexStats, KBStore


class KnowledgeBase:
    def __init__(self, settings: KBSettings) -> None:
        self.settings = settings
        self.store = KBStore(
            settings.db_path,
            settings.docs_dir,
            chunk_chars=settings.chunk_chars,
            chunk_overlap=settings.chunk_overlap,
        )
        self._lock = threading.RLock()
        self._index = BM25Index([])
        self.refresh()

    @property
    def chunk_count(self) -> int:
        return len(self._index)

    def refresh(self) -> int:
        chunks = self.store.load_chunks()
        index = BM25Index(chunks)
        with self._lock:
            self._index = index
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        with self._lock:
            return self._index.search(query, top_k or self.settings.top_k)

    def ingest(self, path: Path) -> DocumentInfo:
        info = self.store.ingest(Path(path))
        self.refresh()
        return info

    def reindex(self) -> IndexStats:
        stats = self.store.reindex()
        stats.chunks = self.refresh()
        return stats

    def documents(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    def delete(self, doc_id: int, *, delete_file: bool = True) -> bool:
        deleted = self.store.delete_document(doc_id, delete_file=delete_file)
        if deleted:
            self.refresh()
        return deleted

    def open_document(self, source: str, max_lines: int | None = None) -> dict[str, object]:
        document = self.store.document_text(source)
        lines = str(document["content"]).splitlines()
        limit = max(1, min(int(max_lines or self.settings.open_doc_lines), 1_000))
        return {
            "source": document["source"],
            "title": document["title"],
            "start_line": 1,
            "total_lines": len(lines),
            "truncated": len(lines) > limit,
            "content": "\n".join(lines[:limit]),
        }
