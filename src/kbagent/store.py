"""SQLite 存储:文档 + 切块。重启不丢索引,新增文档可增量更新。"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .chunking import chunk_markdown, chunk_text
from .config import MAX_FILE_BYTES, SUPPORTED_SUFFIXES
from .loaders import load_sections

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id       INTEGER PRIMARY KEY,
    path     TEXT NOT NULL UNIQUE,
    title    TEXT NOT NULL,
    size     INTEGER NOT NULL,
    mtime    REAL NOT NULL,
    added_at TEXT NOT NULL,
    content  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord        INTEGER NOT NULL,
    text       TEXT NOT NULL,
    page       INTEGER,
    start_line INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


@dataclass(slots=True)
class DocumentInfo:
    id: int
    source: str
    title: str
    size: int
    mtime: float
    chunk_count: int


@dataclass(slots=True)
class ChunkRecord:
    id: int
    doc_id: int
    source: str
    title: str
    text: str
    start_line: int
    page: int | None = None


@dataclass(slots=True)
class IndexStats:
    indexed: int = 0
    removed: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)


class KBStore:
    def __init__(
        self,
        db_path: Path,
        docs_dir: Path,
        *,
        chunk_chars: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self.db_path = db_path
        self.docs_dir = docs_dir
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_schema(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(SCHEMA)

    # -- 写入 -------------------------------------------------------------
    def ingest(self, path: Path) -> DocumentInfo:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"不支持的文件类型: {path.suffix}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"文件太大({size} 字节),上限 {MAX_FILE_BYTES}")

        sections = load_sections(path)
        is_markdown = path.suffix.lower() in {".md", ".markdown"}
        rows: list[tuple[str, int | None, int]] = []
        for section in sections:
            chunks = (
                chunk_markdown(section.text, max_chars=self.chunk_chars, overlap=self.chunk_overlap)
                if is_markdown
                else chunk_text(
                    section.text, max_chars=self.chunk_chars, overlap=self.chunk_overlap
                )
            )
            for chunk in chunks:
                rows.append((chunk.text, section.page, chunk.start_line))
        if not rows:
            raise ValueError("没有可索引的文本")

        content = "\n\n".join(section.text.strip() for section in sections if section.text.strip())
        source = self._display_path(path)
        now = datetime.now(UTC).isoformat(timespec="seconds")

        with closing(self._connect()) as connection, connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE path = ?", (source,)
            ).fetchone()
            if existing:
                doc_id = int(existing["id"])
                connection.execute(
                    "UPDATE documents SET title=?, size=?, mtime=?, added_at=?, content=? "
                    "WHERE id=?",
                    (path.name, size, path.stat().st_mtime, now, content, doc_id),
                )
                connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            else:
                cursor = connection.execute(
                    "INSERT INTO documents(path, title, size, mtime, added_at, content) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (source, path.name, size, path.stat().st_mtime, now, content),
                )
                doc_id = int(cursor.lastrowid or 0)

            connection.executemany(
                "INSERT INTO chunks(doc_id, ord, text, page, start_line) VALUES (?, ?, ?, ?, ?)",
                [
                    (doc_id, order, text, page, start_line)
                    for order, (text, page, start_line) in enumerate(rows)
                ],
            )

        return DocumentInfo(
            id=doc_id,
            source=source,
            title=path.name,
            size=size,
            mtime=path.stat().st_mtime,
            chunk_count=len(rows),
        )

    def delete_document(self, doc_id: int, *, delete_file: bool = True) -> bool:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT path FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                return False
            source = str(row["path"])
            connection.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

        if delete_file:
            target = (self.docs_dir / source).resolve()
            if target.is_relative_to(self.docs_dir) and target.is_file():
                target.unlink()
        return True

    def reindex(self) -> IndexStats:
        stats = IndexStats()
        keep: set[str] = set()

        for path in sorted(self.docs_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            relative = path.relative_to(self.docs_dir)
            if any(part.startswith(".") for part in relative.parts):
                continue
            try:
                info = self.ingest(path)
            except Exception as exc:  # noqa: BLE001 - 单份文档失败要能继续
                stats.errors.append(f"{relative}: {exc}")
                continue
            keep.add(info.source)
            stats.indexed += 1

        for document in self.list_documents():
            if document.source in keep:
                continue
            with closing(self._connect()) as connection, connection:
                connection.execute("DELETE FROM documents WHERE id = ?", (document.id,))
            stats.removed += 1

        stats.chunks = self.count_chunks()
        return stats

    # -- 读取 -------------------------------------------------------------
    def list_documents(self) -> list[DocumentInfo]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT d.id, d.path, d.title, d.size, d.mtime, COUNT(c.id) AS chunk_count "
                "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
                "GROUP BY d.id ORDER BY d.path"
            ).fetchall()
        return [
            DocumentInfo(
                id=int(row["id"]),
                source=str(row["path"]),
                title=str(row["title"]),
                size=int(row["size"]),
                mtime=float(row["mtime"]),
                chunk_count=int(row["chunk_count"]),
            )
            for row in rows
        ]

    def load_chunks(self) -> list[ChunkRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT c.id, c.doc_id, c.text, c.page, c.start_line, "
                "d.path AS source, d.title "
                "FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"
            ).fetchall()
        return [
            ChunkRecord(
                id=int(row["id"]),
                doc_id=int(row["doc_id"]),
                source=str(row["source"]),
                title=str(row["title"]),
                text=str(row["text"]),
                page=row["page"],
                start_line=int(row["start_line"]),
            )
            for row in rows
        ]

    def count_chunks(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0

    def document_text(self, source: str) -> dict[str, object]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, path, title, content FROM documents WHERE path = ?", (source,)
            ).fetchone()
            if row is None:
                matches = connection.execute(
                    "SELECT id, path, title, content FROM documents WHERE title = ? OR path LIKE ?",
                    (source, f"%{source}"),
                ).fetchall()
                if not matches:
                    raise ValueError(f"找不到文档: {source}")
                if len(matches) > 1:
                    options = ", ".join(str(item["path"]) for item in matches)
                    raise ValueError(f"匹配到多份文档,请写完整路径: {options}")
                row = matches[0]
        return {
            "id": int(row["id"]),
            "source": str(row["path"]),
            "title": str(row["title"]),
            "content": str(row["content"]),
        }

    # -- 内部 -------------------------------------------------------------
    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.docs_dir))
        except ValueError:
            return path.name
