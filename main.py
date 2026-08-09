from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
import observability
import rag

app = FastAPI(title="Enterprise Knowledge Assistant", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
observability.setup(app)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    role: str = Field(default="all", max_length=50)


@app.get("/health")
def health():
    return {"status": "ok", "index": config.SEARCH_INDEX, "deployment": config.OPENAI_CHAT_DEPLOYMENT}


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        return rag.answer(
            req.question.strip(),
            history=[t.model_dump() for t in req.history],
            role=req.role,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG pipeline error: {type(exc).__name__}") from exc


@app.post("/api/chat")
def chat(req: AskRequest):
    return StreamingResponse(
        rag.stream_answer(
            req.question.strip(),
            history=[t.model_dump() for t in req.history],
            role=req.role,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_static = Path(__file__).parent / "static"
if _static.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_static, html=True), name="ui")
