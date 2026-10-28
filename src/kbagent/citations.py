"""引用登记簿:把检索命中去重并编号,保证模型写的 [n] 和 UI 展示一致。"""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval import Hit, query_tokens


def _snippet(text: str, query: str = "", limit: int = 240) -> str:
    flat = " ".join(text.split())
    if query:
        lower = flat.lower()
        positions = [lower.find(term) for term in query_tokens(query) if len(term) >= 2]
        positions = [position for position in positions if position >= 0]
        if positions:
            position = min(positions)
            start = max(0, position - 80)
            end = min(len(flat), start + limit)
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(flat) else ""
            return prefix + flat[start:end] + suffix
    return flat if len(flat) <= limit else flat[:limit] + "…"


@dataclass(slots=True)
class Citation:
    cite: int
    chunk_id: int
    source: str
    title: str
    location: str
    snippet: str

    def to_dict(self) -> dict[str, object]:
        return {
            "cite": self.cite,
            "source": self.source,
            "title": self.title,
            "location": self.location,
            "snippet": self.snippet,
        }


class CitationRegistry:
    def __init__(self) -> None:
        self._items: list[Citation] = []
        self._by_chunk: dict[int, int] = {}

    def clear(self) -> None:
        self._items.clear()
        self._by_chunk.clear()

    def add(self, hit: Hit, query: str = "") -> int:
        known = self._by_chunk.get(hit.chunk_id)
        if known is not None:
            return known
        cite = len(self._items) + 1
        self._by_chunk[hit.chunk_id] = cite
        self._items.append(
            Citation(
                cite=cite,
                chunk_id=hit.chunk_id,
                source=hit.source,
                title=hit.title,
                location=hit.location,
                snippet=_snippet(hit.text, query),
            )
        )
        return cite

    def items(self) -> list[Citation]:
        return list(self._items)

    def as_dicts(self) -> list[dict[str, object]]:
        return [item.to_dict() for item in self._items]

    def __len__(self) -> int:
        return len(self._items)
