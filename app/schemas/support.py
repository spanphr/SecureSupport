"""Contracts for the first, single-turn support workflow."""
from typing import Literal
from pydantic import Field
from app.schemas.domain import Record

Route = Literal["direct", "retrieval", "tool", "out_of_scope"]
Status = Literal["answered", "action_unavailable", "out_of_scope", "insufficient_context", "error"]


class SupportRequest(Record):
    message: str = Field(min_length=1, max_length=4000)


class RoutingDecision(Record):
    route: Route


class GroundedAnswer(Record):
    answer: str = Field(min_length=1)
    source_ids: list[str]
    supported: bool


class SupportResponse(Record):
    route: Route | None
    status: Status
    answer: str
    source_ids: list[str]
    action_performed: Literal[False] = False
