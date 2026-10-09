"""Retrieval results expose sources and distances, not probability scores."""
from pydantic import Field
from app.schemas.domain import Record


class RetrievalHit(Record):
    article_id: str
    title: str
    content: str
    distance: float = Field(allow_inf_nan=False)


class RetrievalResult(Record):
    query: str
    hits: list[RetrievalHit]
