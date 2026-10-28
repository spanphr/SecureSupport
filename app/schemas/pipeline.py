"""Contracts for the end-to-end support pipeline: identity, tools, escalation, response."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from app.schemas.decision import Action, Category
from app.schemas.domain import Record

PipelineStatus = Literal[
    "answered",            # grounded policy or account answer
    "action_completed",    # a write tool confirmed success
    "clarification_required",
    "refused",             # out of scope, restricted, or unauthorized
    "unverified_action",   # answer claimed an action without tool confirmation
    "escalated",           # human handoff created and persisted
    "insufficient_context",
    "error",
]


class SessionContext(Record):
    """Trusted server-side identity.

    In a deployed system this is built from a verified session. It must never be
    parsed from the customer message or from LLM output.
    """

    customer_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


class ConversationTurn(Record):
    role: Literal["customer", "agent"]
    content: str = Field(min_length=1, max_length=4000)


class PipelineRequest(Record):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ConversationTurn] = Field(default_factory=list, max_length=20)
    # Confirmation for state-changing tools other than escalation. Supplied by the
    # application after an explicit customer confirmation, never inferred by the model.
    confirm_write: bool = False


class LLMUsage(Record):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class GroundedAnswer(Record):
    """Composition result. `supported` is a model claim that code verifies."""

    answer: str = Field(min_length=1, max_length=4000)
    source_ids: list[str] = Field(default_factory=list)
    supported: bool


ToolOutcomeCode = Literal[
    "ok",
    "unknown_tool",
    "not_authorized",
    "confirmation_required",
    "invalid_arguments",
    "not_found",
    "execution_failed",
]


class ToolOutcome(Record):
    """Structured tool result. Failures are values, not exceptions."""

    tool_name: str
    code: ToolOutcomeCode
    message: str = Field(min_length=1, max_length=300)
    payload: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_payload(self):
        if self.code == "ok" and self.payload is None:
            raise ValueError("Successful outcomes require a payload.")
        if self.code != "ok" and self.payload is not None:
            raise ValueError("Failed outcomes must not carry a payload.")
        return self

    @property
    def ok(self) -> bool:
        return self.code == "ok"


class ToolCallRecord(Record):
    tool_name: str
    arguments: dict[str, str]
    outcome: ToolOutcome
    duration_ms: float = Field(ge=0)


class EvidenceRef(Record):
    article_id: str
    title: str
    rank: int = Field(ge=1)
    distance: float


class ToolAttempt(Record):
    tool_name: str
    code: ToolOutcomeCode


class EscalationReason(StrEnum):
    CUSTOMER_REQUESTED = "customer_requested"
    MISSING_KNOWLEDGE = "missing_knowledge"
    UNRESOLVED_AMBIGUITY = "unresolved_ambiguity"
    AUTHORIZATION_SENSITIVE = "authorization_sensitive"
    TOOL_FAILURE = "tool_failure"
    REQUIRES_HUMAN_REVIEW = "requires_human_review"


class EscalationHandoff(Record):
    """Structured handoff a human agent can act on without replaying the chat."""

    reason: EscalationReason
    category: Category
    summary: str = Field(min_length=1, max_length=600)
    detected_issue: str = Field(min_length=1, max_length=300)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    tools_attempted: list[ToolAttempt] = Field(default_factory=list)
    trace_id: str
    requires_human_review: bool = True
    created_at: datetime


class PipelineResponse(Record):
    trace_id: str
    status: PipelineStatus
    answer: str
    category: Category | None = None
    action: Action | None = None
    source_ids: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    action_performed: bool = False
    escalation: EscalationHandoff | None = None
    escalation_ticket_id: str | None = None
    total_latency_ms: float = Field(ge=0)
