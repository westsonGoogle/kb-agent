"""切块:按行累积到上限,带行号,保留一小段重叠。

行号是引用能落地的关键——用户点开来源时要知道看第几行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    text: str
    start_line: int = 1
    page: int | None = None


_HEADING = re.compile(r"^#{1,6}\s+\S")


def chunk_text(
    text: str,
    *,
    max_chars: int = 800,
    overlap: int = 120,
    line_offset: int = 0,
) -> list[Chunk]:
    if max_chars < 100:
        raise ValueError("max_chars 至少 100")
    overlap = max(0, min(overlap, max_chars // 2))

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []

    lines = normalized.split("\n")
    chunks: list[Chunk] = []
    buf: list[tuple[int, str]] = []
    size = 0
    dirty = False

    def emit() -> None:
        nonlocal buf, size, dirty
        body = "\n".join(line for _, line in buf).strip()
        if body:
            chunks.append(Chunk(text=body, start_line=line_offset + buf[0][0]))

        # 保留尾部若干行做重叠,让跨块的句子不被切断
        keep: list[tuple[int, str]] = []
        keep_size = 0
        for item in reversed(buf):
            add = len(item[1]) + 1
            if keep_size + add > overlap:
                break
            keep.insert(0, item)
            keep_size += add
        buf = keep
        size = keep_size
        dirty = False

    for lineno, line in enumerate(lines, start=1):
        # 超长单行(压缩代码、表格)硬切,避免一个块爆掉
        if len(line) > max_chars:
            if buf:
                emit()
            buf, size, dirty = [], 0, False
            for start in range(0, len(line), max_chars):
                piece = line[start : start + max_chars].strip()
                if piece:
                    chunks.append(Chunk(text=piece, start_line=line_offset + lineno))
            continue

        if buf and size + len(line) + 1 > max_chars:
            emit()

        buf.append((lineno, line))
        size += len(line) + 1
        dirty = True

    if buf and dirty:
        emit()
    return chunks


def chunk_markdown(
    text: str,
    *,
    max_chars: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """按标题切段再切块,避免一个块从上一节尾巴开始。

    引用时用户看到的第一行就是标题,比看到半句结构化输出有用得多。
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    sections: list[tuple[int, list[str]]] = []
    start = 1
    buffer: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        if _HEADING.match(line) and buffer:
            sections.append((start, buffer))
            buffer = []
            start = lineno
        buffer.append(line)
    if buffer:
        sections.append((start, buffer))

    chunks: list[Chunk] = []
    for section_start, section_lines in sections:
        section_text = "\n".join(section_lines)
        chunks.extend(
            chunk_text(
                section_text,
                max_chars=max_chars,
                overlap=overlap,
                line_offset=section_start - 1,
            )
        )
    return chunks
