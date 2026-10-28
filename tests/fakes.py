from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

from agentlab.llm import LLMResponse, StreamEvent, ToolCall, Usage


def _usage() -> Usage:
    return Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def text_script(text: str) -> Iterator[StreamEvent]:
    used = _usage()
    yield StreamEvent("text", text=text)
    yield StreamEvent("usage", usage=used)
    yield StreamEvent("final", response=LLMResponse(content=text, usage=used, finish_reason="stop"))


def tool_script(*calls: ToolCall, content: str = "") -> Iterator[StreamEvent]:
    response = LLMResponse(
        content=content,
        tool_calls=list(calls),
        usage=_usage(),
        finish_reason="tool_calls",
    )
    yield StreamEvent("final", response=response)


class FakeLLM:
    def __init__(self, scripts: list[Iterator[StreamEvent]]) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, **kwargs})
        if not self.scripts:
            raise AssertionError("FakeLLM 没有更多脚本了")
        yield from self.scripts.pop(0)
