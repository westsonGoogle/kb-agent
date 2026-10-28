from __future__ import annotations

import pytest

from kbagent.chunking import chunk_markdown, chunk_text


def test_empty_text() -> None:
    assert chunk_text("   \n\n ") == []


def test_single_chunk_with_line_numbers() -> None:
    chunks = chunk_text("第一行\n第二行\n第三行", max_chars=100)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].text == "第一行\n第二行\n第三行"


def test_chunks_respect_size_and_overlap() -> None:
    text = "\n".join(f"第 {index} 行,这里是一些用来凑长度内容。" for index in range(1, 61))
    chunks = chunk_text(text, max_chars=220, overlap=60)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 260 for chunk in chunks)
    assert chunks[0].start_line == 1

    # 行号递增,且相邻块有重叠(后一块起始行 <= 前一块结束行 + 1)
    for previous, current in zip(chunks, chunks[1:]):
        previous_end = previous.start_line + previous.text.count("\n")
        assert current.start_line <= previous_end + 1


def test_overlap_content_is_preserved() -> None:
    lines = [f"line-{index}" for index in range(1, 31)]
    chunks = chunk_text("\n".join(lines), max_chars=120, overlap=50)
    joined = "\n".join(chunk.text for chunk in chunks)
    for line in lines:
        assert line in joined


def test_long_single_line_is_hard_split() -> None:
    chunks = chunk_text("x" * 500, max_chars=120, overlap=20)
    assert len(chunks) >= 4
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert chunks[0].start_line == 1


def test_invalid_max_chars() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=50)


def test_markdown_chunks_start_at_headings() -> None:
    text = "# A\n\nintro\n\n## B\n\nbody b\n\n## C\n\nbody c"
    chunks = chunk_markdown(text, max_chars=300, overlap=20)

    assert [chunk.text.splitlines()[0] for chunk in chunks] == ["# A", "## B", "## C"]
    assert [chunk.start_line for chunk in chunks] == [1, 5, 9]


def test_markdown_long_section_is_split() -> None:
    body = "\n".join(f"第 {index} 行内容,用来把这一节撑长。" for index in range(1, 30))
    text = "## 长章节\n\n" + body
    chunks = chunk_markdown(text, max_chars=200, overlap=40)

    assert len(chunks) > 1
    assert all(chunk.start_line >= 1 for chunk in chunks)
    assert chunks[0].text.startswith("## 长章节")
