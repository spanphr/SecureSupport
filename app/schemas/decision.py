"""Structured agent decisions with deterministic action/tool consistency rules.

The model proposes a decision; this module decides whether that decision is
internally coherent. Authority (who the customer is, which tools exist, whether a
write is confirmed) never comes from these fields - see app.guardrails.
"""

from enum import StrEnum

from pydantic import Field, model_validator

from app.schemas.domain import Record


class Category(StrEnum):
    GREETING = "greeting"
    PAYMENTS = "payments"
    REFUNDS = "refunds"
    ACCOUNT_ACCESS = "account_access"
    SECURITY = "security"
    COMPLAINTS = "complaints"
    DATA_PRIVACY = "data_privacy"
    SUPPORT_SCOPE = "support_scope"
    OUT_OF_SCOPE = "out_of_scope"


class Action(StrEnum):
    RESPOND = "respond"
    ANSWER_POLICY = "answer_policy"
    LOOKUP_ACCOUNT = "lookup_account"
    LOOKUP_PAYMENT = "lookup_payment"
    LOOKUP_TICKET = "lookup_ticket"
    CREATE_TICKET = "create_ticket"
    CLARIFY = "clarify"
    ESCALATE = "escalate"
    REFUSE = "refuse"


# The only mapping from an action to a tool. A decision naming any other tool, or
# naming a tool without the matching action, is rejected before execution.
TOOL_FOR_ACTION: dict[Action, str] = {
    Action.LOOKUP_ACCOUNT: "get_account_status",
    Action.LOOKUP_PAYMENT: "get_payment_status",
    Action.LOOKUP_TICKET: "get_ticket_status",
    Action.CREATE_TICKET: "create_support_ticket",
}

RETRIEVAL_ACTIONS = frozenset({Action.ANSWER_POLICY})
TOOL_ACTIONS = frozenset(TOOL_FOR_ACTION)
WRITE_ACTIONS = frozenset({Action.CREATE_TICKET})


class AgentDecision(Record):
    """A single structured decision. No self-reported confidence is trusted."""

    category: Category
    action: Action
    needs_retrieval: bool
    needs_tool: bool
    needs_clarification: bool
    needs_escalation: bool
    tool_name: str | None = Field(default=None, min_length=1, max_length=64)
    tool_arguments: dict[str, str] | None = None
    clarification_question: str | None = Field(default=None, min_length=1, max_length=300)

    @model_validator(mode="after")
    def check_consistency(self):
        # Tool actions must name exactly their mapped tool and supply arguments.
        if self.action in TOOL_ACTIONS:
            if not self.needs_tool:
                raise ValueError("Tool actions require needs_tool=true.")
            if self.tool_name != TOOL_FOR_ACTION[self.action]:
                raise ValueError("tool_name does not match the selected action.")
            if self.tool_arguments is None:
                raise ValueError("Tool actions require tool_arguments (possibly empty).")
        else:
            if self.needs_tool or self.tool_name is not None or self.tool_arguments is not None:
                raise ValueError("Only tool actions may request a tool.")

        if self.action is Action.ANSWER_POLICY and not self.needs_retrieval:
            raise ValueError("Policy answers require retrieval.")
        if self.needs_retrieval and self.action not in RETRIEVAL_ACTIONS | TOOL_ACTIONS:
            # Retrieval is only meaningful for policy answers and tool-backed answers.
            raise ValueError("needs_retrieval is only valid for policy or tool actions.")

        if self.action is Action.CLARIFY:
            if not self.needs_clarification or self.clarification_question is None:
                raise ValueError("Clarification actions require a question.")
        elif self.needs_clarification or self.clarification_question is not None:
            raise ValueError("Only clarification actions may ask a question.")

        if self.needs_escalation != (self.action is Action.ESCALATE):
            raise ValueError("needs_escalation must match the selected action.")

        if self.action is Action.REFUSE and self.category is Category.GREETING:
            raise ValueError("A greeting cannot be a refusal.")
        if self.action is Action.RESPOND and self.category not in (
            Category.GREETING, Category.SUPPORT_SCOPE
        ):
            # Small talk is answered deterministically; substantive questions need evidence.
            raise ValueError("respond is only valid for greetings and capability questions.")
        if self.action is Action.ANSWER_POLICY and self.category is Category.GREETING:
            raise ValueError("A greeting does not need policy retrieval.")
        return self

    @property
    def is_write(self) -> bool:
        return self.action in WRITE_ACTIONS
