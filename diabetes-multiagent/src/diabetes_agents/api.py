from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .orchestrator import Orchestrator

app = FastAPI(title="Diabetes Multi-Agent Assistant", version="0.1.0")
orchestrator = Orchestrator()


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=2000)
    include_trace: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    result = asdict(orchestrator.handle(req.session_id, req.message))
    if not req.include_trace:
        result.pop("trace")
    return result


@app.get("/escalations")
def escalations():
    """Clinician queue. In production, protect with authentication and audit logging."""
    return orchestrator.notifier.list_tickets()
