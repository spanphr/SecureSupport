"""Explicit, per-request pipeline state. No hidden memory between requests."""

from typing import Literal, TypedDict

from app.guardrails.input import InputAssessment
from app.observability import TraceBuilder
from app.schemas.decision import AgentDecision
from app.schemas.pipeline import (
    ConversationTurn,
    EscalationHandoff,
    EscalationReason,
    PipelineRequest,
    PipelineStatus,
    SessionContext,
    ToolCallRecord,
)
from app.schemas.retrieval import RetrievalHit

NextNode = Literal[
    "refuse", "clarify", "respond", "decide", "retrieve", "compose",
    "tool", "escalate", "check_output", "failure",
]


class PipelineState(TypedDict, total=False):
    request: PipelineRequest
    session: SessionContext
    history: list[ConversationTurn]
    trace: TraceBuilder
    assessment: InputAssessment
    decision: AgentDecision
    hits: list[RetrievalHit]
    tool_calls: list[ToolCallRecord]
    answer: str
    source_ids: list[str]
    status: PipelineStatus
    escalation: EscalationHandoff | None
    escalation_reason: EscalationReason | None
    escalation_ticket_id: str | None
    action_performed: bool
    next: NextNode
    failed: bool
