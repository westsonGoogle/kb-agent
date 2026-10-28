"""配置:模型沿用 agent-lab 的 Settings,知识库参数单独管理。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"}
MAX_FILE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class KBSettings:
    data_dir: Path = Path("data")
    docs_dir: Path = Path("data/docs")
    db_path: Path = Path("data/kb.db")
    top_k: int = 6
    chunk_chars: int = 800
    chunk_overlap: int = 120
    open_doc_lines: int = 200

    @classmethod
    def from_env(cls) -> KBSettings:
        load_dotenv(find_dotenv(usecwd=True))
        data_dir = Path(os.getenv("KB_DATA_DIR", "data")).expanduser().resolve()
        docs_dir = Path(os.getenv("KB_DOCS_DIR", str(data_dir / "docs"))).expanduser().resolve()
        db_path = Path(os.getenv("KB_DB_PATH", str(data_dir / "kb.db"))).expanduser().resolve()
        return cls(
            data_dir=data_dir,
            docs_dir=docs_dir,
            db_path=db_path,
            top_k=int(os.getenv("KB_TOP_K", "6")),
            chunk_chars=int(os.getenv("KB_CHUNK_CHARS", "800")),
            chunk_overlap=int(os.getenv("KB_CHUNK_OVERLAP", "120")),
            open_doc_lines=int(os.getenv("KB_OPEN_DOC_LINES", "200")),
        )
