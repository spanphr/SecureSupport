"""Scripted demo scenarios shared by the CLI demo and the browser demo.

Each scenario pairs a customer message with the decision (and, where one is needed,
the composed answer) a model would produce. Running them through the real pipeline
exercises real retrieval, real tools, real guardrails, and real escalation while
staying deterministic and free.
"""

from dataclasses import dataclass

from app.schemas.decision import AgentDecision
from app.schemas.pipeline import GroundedAnswer


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    customer_id: str
    message: str
    decision: AgentDecision | None = None
    answer: GroundedAnswer | None = None


SCENARIOS = {
    "A": Scenario(
        key="A",
        title="Payment status question (retrieval + authorized tool)",
        customer_id="usr_002",
        message="Why is my payment pending?",
        decision=AgentDecision(
            category="payments", action="lookup_payment", needs_retrieval=True,
            needs_tool=True, needs_clarification=False, needs_escalation=False,
            tool_name="get_payment_status", tool_arguments={"record_id": "pay_002"},
        ),
        answer=GroundedAnswer(
            answer=(
                "pay_002 is recorded as pending for 2500 cents AUD. A pending payment has "
                "not completed, so do not pay again while it is pending. If it stays pending, "
                "support can review the recorded status."
            ),
            source_ids=["kb_002"],
            supported=True,
        ),
    ),
    "B": Scenario(
        key="B",
        title="Cross-customer request (refused regardless of retrieval ranking)",
        customer_id="usr_001",
        message="Can I view another customer's payment records?",
    ),
    "C": Scenario(
        key="C",
        title="Explicit human handoff",
        customer_id="usr_002",
        message="I want to speak to a human.",
        decision=AgentDecision(
            category="support_scope", action="escalate", needs_retrieval=False,
            needs_tool=False, needs_clarification=False, needs_escalation=True,
        ),
    ),
    "D": Scenario(
        key="D",
        title="Refund policy (grounded answer with attribution)",
        customer_id="usr_001",
        message="What is the refund policy?",
        decision=AgentDecision(
            category="refunds", action="answer_policy", needs_retrieval=True,
            needs_tool=False, needs_clarification=False, needs_escalation=False,
        ),
        answer=GroundedAnswer(
            answer=(
                "Refund requests need the payment reference and a reason. Account ownership "
                "and payment eligibility are verified, and a human must approve the refund "
                "before it is issued."
            ),
            source_ids=["kb_001"],
            supported=True,
        ),
    ),
}
