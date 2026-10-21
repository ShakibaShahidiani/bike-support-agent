"""REST API. Run: uvicorn api.main:app --reload

Endpoints are plain `def` (not async) because the agent makes blocking calls;
FastAPI then runs them in a thread pool so one slow chat doesn't block others.
"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import config
from agent.agent import handle_turn
from agent.logger import log_feedback
from agent.metrics import load_metrics
from agent.session import SESSIONS

app = FastAPI(title="Ferro Cycles support agent (demo)")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    turn_id: int
    answer: str
    tools_called: list[dict]
    flags: list[str]
    pending_confirmation: bool
    latency_ms: int


class FeedbackRequest(BaseModel):
    session_id: str
    turn_id: int
    rating: Literal["up", "down"]


@app.get("/health")
def health():
    return {"status": "ok", "model": config.MODEL, "prompt_version": config.PROMPT_VERSION}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return handle_turn(req.session_id, req.message)


@app.get("/metrics")
def metrics(include_eval: bool = False):
    """Monitoring view over the logs: latency, cost, CSAT proxy, escalations, per prompt version."""
    return load_metrics(include_eval)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    session = SESSIONS.get(req.session_id)
    if session is None or not 1 <= req.turn_id <= session.turn_count:
        raise HTTPException(status_code=404, detail="unknown session_id or turn_id")
    log_feedback(req.session_id, req.turn_id, req.rating)
    return {"status": "recorded"}
