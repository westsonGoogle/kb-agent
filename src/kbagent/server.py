"""FastAPI:Web 界面 + SSE 聊天 + 文档上传/删除/重建索引。"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentlab.config import Settings
from agentlab.loop import Agent
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import build_agent
from .citations import CitationRegistry
from .config import MAX_FILE_BYTES, SUPPORTED_SUFFIXES, KBSettings
from .kb import KnowledgeBase

AgentFactory = Callable[[KnowledgeBase, CitationRegistry], Agent]


@dataclass(slots=True)
class Session:
    agent: Agent
    citations: CitationRegistry


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(default=None, max_length=64)


def create_app(
    *,
    kb: KnowledgeBase | None = None,
    settings: Settings | None = None,
    agent_factory: AgentFactory | None = None,
) -> FastAPI:
    knowledge = kb or KnowledgeBase(KBSettings.from_env())
    model_settings = settings or Settings.from_env()
    factory: AgentFactory = agent_factory or (
        lambda base, citations: build_agent(base, citations, model_settings)
    )

    sessions: dict[str, Session] = {}
    app = FastAPI(title="kb-agent", version="0.1.0")
    app.state.kb = knowledge
    app.state.sessions = sessions
    app.state.settings = model_settings

    index_html = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return index_html

    @app.get("/favicon.ico")
    def favicon() -> HTMLResponse:
        return HTMLResponse(status_code=204)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": model_settings.model,
            "documents": len(knowledge.documents()),
            "chunks": knowledge.chunk_count,
        }

    @app.get("/api/documents")
    def documents() -> dict[str, Any]:
        return {
            "documents": [asdict(item) for item in knowledge.documents()],
            "chunks": knowledge.chunk_count,
            "docs_dir": str(knowledge.settings.docs_dir),
        }

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        name = Path(file.filename or "").name
        if not name or name.startswith("."):
            raise HTTPException(status_code=400, detail="文件名不合法")
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=400, detail="只支持 md / txt / pdf / docx")

        data = await file.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="文件超过 20MB")

        destination = knowledge.settings.docs_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        try:
            info = knowledge.ingest(destination)
        except Exception as exc:  # noqa: BLE001
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return asdict(info)

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: int) -> dict[str, bool]:
        if not knowledge.delete(doc_id):
            raise HTTPException(status_code=404, detail="文档不存在")
        return {"deleted": True}

    @app.post("/api/reindex")
    def reindex() -> dict[str, Any]:
        return asdict(knowledge.reindex())

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        if req.session_id:
            session_id = req.session_id
            session = sessions.get(session_id)
            if session is None:
                citations = CitationRegistry()
                session = sessions[session_id] = Session(
                    agent=factory(knowledge, citations), citations=citations
                )
        else:
            session_id = uuid.uuid4().hex[:8]
            citations = CitationRegistry()
            session = sessions[session_id] = Session(
                agent=factory(knowledge, citations), citations=citations
            )

        session.citations.clear()

        def event_stream() -> Iterator[str]:
            yield _sse({"type": "session", "session_id": session_id})
            try:
                for event in session.agent.stream(req.message):
                    yield _sse(event.to_dict())
                    if event.type == "done":
                        yield _sse({"type": "sources", "sources": session.citations.as_dicts()})
            except Exception as exc:  # noqa: BLE001 - 单次对话失败不应打死服务
                yield _sse({"type": "error", "error": str(exc)})
            yield _sse({"type": "end"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, bool]:
        return {"deleted": sessions.pop(session_id, None) is not None}

    return app


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
