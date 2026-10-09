"""Shared state passed between LangGraph nodes; no persistence yet."""
from typing import TypedDict
from app.schemas.retrieval import RetrievalHit
from app.schemas.support import Route, Status


class SupportState(TypedDict, total=False):
    message: str
    route: Route | None
    hits: list[RetrievalHit]
    answer: str
    source_ids: list[str]
    status: Status
    failed: bool
