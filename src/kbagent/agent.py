"""Agent 组装:system prompt + 三个检索工具。

Agent 循环本身来自 agent-lab(已开源),这里只负责「知识库」这一层。
"""

from __future__ import annotations

import json

from agentlab import Agent, LLMClient, Tool, ToolRegistry
from agentlab.config import Settings

from .citations import CitationRegistry
from .kb import KnowledgeBase

SYSTEM_PROMPT = """你是「个人知识库助手」,只能依据本地文档库回答问题。

必须遵守:
1. 回答事实性问题前,先调用 search_kb 检索;问题复杂时换 2-3 组关键词多次检索。
2. 只能使用检索结果或 open_document 返回的内容。禁止用训练记忆补充,禁止编造。
3. 每个结论后标注来源,格式 [n],n 是工具返回的 cite 编号。多个来源写 [1][3]。
   只引用真正支持该结论的片段,不要为了凑引用而引用。
4. 资料不足时直接说「资料里没有提到」,并说明还需要哪类资料。不要猜。
5. 用中文回答:先给结论,再给要点。不要复述检索过程,也不要输出 JSON。
6. 用户问「知识库有哪些文档」时,调用 list_documents。

引用示例:
「prompt cache 要求前缀完全一致 [2],动态时间戳会破坏命中 [2][4]。」"""


def build_agent(
    kb: KnowledgeBase,
    citations: CitationRegistry,
    settings: Settings,
    *,
    top_k: int | None = None,
) -> Agent:
    default_top_k = top_k or kb.settings.top_k

    def search_kb(query: str, top_k: int | None = None) -> str:
        hits = kb.search(query, top_k or default_top_k)
        if not hits:
            return json.dumps(
                {
                    "query": query,
                    "results": [],
                    "hint": "没有命中。换同义词、更短的关键词或英文术语再试。",
                },
                ensure_ascii=False,
            )

        results = []
        for hit in hits:
            results.append(
                {
                    "cite": citations.add(hit, query),
                    "source": hit.source,
                    "location": hit.location,
                    "score": round(hit.score, 2),
                    "text": hit.text[:900],
                }
            )
        return json.dumps(
            {"query": query, "results": results, "count": len(results)},
            ensure_ascii=False,
        )

    def open_document(source: str, max_lines: int | None = None) -> str:
        return json.dumps(kb.open_document(source, max_lines), ensure_ascii=False)

    def list_documents() -> str:
        documents = kb.documents()
        return json.dumps(
            {
                "documents": [
                    {"source": item.source, "title": item.title, "chunks": item.chunk_count}
                    for item in documents
                ],
                "total": len(documents),
            },
            ensure_ascii=False,
        )

    tools = ToolRegistry(
        [
            Tool(
                name="search_kb",
                description=(
                    "检索本地知识库,返回最相关的文档片段。每条结果带 cite 编号,"
                    "回答时必须用 [cite] 标注来源。一次没搜到就换关键词再搜。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "检索词。只放 2-5 个关键词(中英文都可以),不要放整句或疑问句"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"返回条数,默认 {default_top_k}",
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                },
                func=search_kb,
                example={"query": "prompt caching 命中", "top_k": 5},
            ),
            Tool(
                name="open_document",
                description=(
                    "打开一份文档的正文(按行切片)。检索片段不够用时再调用,"
                    "source 必须来自检索结果或 list_documents。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "文档相对路径,例如 03-uv-fastapi.md",
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "最多读取多少行,默认 200",
                            "minimum": 1,
                            "maximum": 1000,
                        },
                    },
                    "required": ["source"],
                },
                func=open_document,
                example={"source": "03-uv-fastapi.md", "max_lines": 100},
            ),
            Tool(
                name="list_documents",
                description="列出知识库中的全部文档及切块数量。",
                parameters={"type": "object", "properties": {}, "required": []},
                func=list_documents,
            ),
        ]
    )

    return Agent(
        llm=LLMClient(settings),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        max_steps=6,
    )
