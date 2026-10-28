"""Pipeline nodes.

Each node does one thing and records what it did on the trace. Model calls are
isolated behind `SupportModel`, retrieval behind `PolicySearcher`, and state
changes behind `ToolExecutor`, so the whole workflow runs offline in tests.
"""

import json
import time
from datetime import datetime, timezone
from typing import Protocol

from app.agent.model import SupportModel
from app.agent.state import PipelineState
from app.config import Settings
from app.guardrails import input as input_guard
from app.guardrails import output as output_guard
from app.guardrails.input import InputCode
from app.schemas.decision import Action, Category
from app.schemas.pipeline import (
    EscalationHandoff,
    EscalationReason,
    EvidenceRef,
    ToolAttempt,
)
from app.schemas.retrieval import RetrievalResult
from app.tools.registry import WRITE_TOOL_NAMES, ToolExecutor, catalog

ERROR_ANSWER = (
    "I could not complete this request. Please try again. "
    "No record was accessed and no account action was performed."
)
INVALID_ANSWER = "That message could not be processed. Please send between 1 and 4000 characters."
NOT_FOUND_ANSWER = (
    "I could not find that record on your account. Nothing was accessed or changed."
)
CAPABILITY_ANSWER = (
    "Hello. I can explain payment, refund, account recovery, complaint, and privacy policies. "
    "I can also look up your own account, payment, or ticket status and open a support ticket. "
    "I cannot access another customer's records or perform refunds and account changes."
)
CONFIRM_TEMPLATE = (
    "I can open a support ticket titled '{subject}'. That creates a new record on your account. "
    "Confirm and I will create it."
)

REFUSAL_ANSWERS = {
    InputCode.CROSS_CUSTOMER: (
        "I can only access records that belong to the authenticated account. "
        "I cannot look up or share another customer's information."
    ),
    InputCode.INJECTION_SIGNAL: (
        "I can't share internal instructions or change how this assistant works. "
        "I can help with account, payment, and policy questions instead."
    ),
    InputCode.SECRET_REQUEST: (
        "I can't share credentials, configuration, or internal instructions. "
        "I can help with account, payment, and policy questions instead."
    ),
}
PRIVACY_REFUSAL = (
    "I can't help with that request. I only access the authenticated customer's own records."
)
RESTRICTED_REFUSAL = (
    "I can't perform refunds, cancellations, account deletions, or MFA changes. "
    "No action was taken. I can explain the policy and open a support ticket for a human to review."
)
SCOPE_REFUSAL = (
    "I can help with account, payment, complaint, and privacy support only. "
    "I can't provide unrelated advice or internal information."
)

DETECTED_ISSUE = {
    EscalationReason.CUSTOMER_REQUESTED: "Customer asked to speak with a human.",
    EscalationReason.MISSING_KNOWLEDGE: "No approved policy article answered the question.",
    EscalationReason.UNRESOLVED_AMBIGUITY: "The request remained ambiguous after clarification.",
    EscalationReason.AUTHORIZATION_SENSITIVE: "The request needs verification that chat cannot provide.",
    EscalationReason.TOOL_FAILURE: "A backend operation did not complete successfully.",
    EscalationReason.REQUIRES_HUMAN_REVIEW: "The request needs human review.",
}

ESCALATION_ANSWERS = {
    EscalationReason.CUSTOMER_REQUESTED: (
        "I've opened support ticket {ticket} and passed your request to a human support agent."
    ),
    EscalationReason.MISSING_KNOWLEDGE: (
        "I couldn't find an approved policy that answers that, so I've opened support ticket "
        "{ticket} for a human to follow up. No account action was performed."
    ),
    EscalationReason.AUTHORIZATION_SENSITIVE: (
        "This needs verification that chat cannot provide, so I've opened support ticket "
        "{ticket}. No account change was performed."
    ),
    EscalationReason.TOOL_FAILURE: (
        "Something went wrong while completing that request, so I've opened support ticket "
        "{ticket}. No change was confirmed."
    ),
    EscalationReason.UNRESOLVED_AMBIGUITY: (
        "I couldn't resolve that safely here, so I've opened support ticket {ticket} "
        "for a human to help."
    ),
    EscalationReason.REQUIRES_HUMAN_REVIEW: (
        "I've opened support ticket {ticket} so a human can review this request."
    ),
}


class PolicySearcher(Protocol):
    def search(self, query: str, top_k: int = 3) -> RetrievalResult: ...


class PipelineNodes:
    def __init__(self, model: SupportModel, searcher: PolicySearcher,
                 executor: ToolExecutor, settings: Settings):
        self.model = model
        self.searcher = searcher
        self.executor = executor
        self.settings = settings

    # --- entry -------------------------------------------------------------
    def validate(self, state: PipelineState) -> dict:
        assessment = input_guard.assess(state["request"].message)
        state["trace"].add_flags(assessment.flags)
        if assessment.ok:
            return {"assessment": assessment, "next": "decide", "failed": False}
        if assessment.code is InputCode.INVALID:
            return {
                "assessment": assessment, "answer": INVALID_ANSWER, "status": "error",
                "source_ids": [], "failed": True, "next": "failure",
            }
        # Injection, secret requests, and cross-customer requests are refused
        # before retrieval, so ranking can never unlock them.
        return {
            "assessment": assessment, "source_ids": [], "tool_calls": [],
            "status": "refused", "failed": False, "next": "refuse",
        }

    def decide(self, state: PipelineState) -> dict:
        started = time.perf_counter()
        try:
            result = self.model.decide(
                message=state["request"].message,
                history=state["history"],
                tools=catalog(),
            )
        except Exception as error:
            state["trace"].add_error(type(error).__name__)
            return {"failed": True, "status": "error", "next": "failure"}

        state["trace"].add_llm(result.usage, (time.perf_counter() - started) * 1000)
        decision = result.value
        state["trace"].set_decision(decision.category.value, decision.action.value)

        if "restricted_operation" in state["trace"].flags and decision.is_write:
            # The model may not route around a policy refusal by opening a ticket.
            return {"decision": decision, "failed": False, "next": "refuse"}

        if decision.needs_tool:
            return {"decision": decision, "failed": False, "next": "tool"}
        if decision.action is Action.RESPOND:
            return {"decision": decision, "failed": False, "next": "respond"}
        if decision.action is Action.CLARIFY:
            return {"decision": decision, "failed": False, "next": "clarify"}
        if decision.action is Action.ESCALATE:
            return {"decision": decision, "escalation_reason": EscalationReason.CUSTOMER_REQUESTED,
                    "failed": False, "next": "escalate"}
        if decision.action is Action.REFUSE:
            return {"decision": decision, "failed": False, "next": "refuse"}
        return {"decision": decision, "failed": False, "next": "retrieve"}

    # --- deterministic answers --------------------------------------------
    def respond(self, state: PipelineState) -> dict:
        return {"answer": CAPABILITY_ANSWER, "status": "answered", "source_ids": []}

    def clarify(self, state: PipelineState) -> dict:
        return {
            "answer": state["decision"].clarification_question,
            "status": "clarification_required",
            "source_ids": [],
        }

    def refuse(self, state: PipelineState) -> dict:
        assessment = state.get("assessment")
        code = assessment.code if assessment else InputCode.OK
        decision = state.get("decision")
        if "restricted_operation" in state["trace"].flags:
            answer = RESTRICTED_REFUSAL
        elif code in REFUSAL_ANSWERS:
            answer = REFUSAL_ANSWERS[code]
        elif decision is not None and decision.category is Category.OUT_OF_SCOPE:
            answer = SCOPE_REFUSAL
        else:
            answer = PRIVACY_REFUSAL
        return {"answer": answer, "status": "refused", "source_ids": [], "tool_calls": []}

    # --- retrieval and composition ----------------------------------------
    def _within_cutoff(self, hits):
        ceiling = self.settings.retrieval_max_distance
        return [hit for hit in hits if ceiling is None or hit.distance <= ceiling]

    def retrieve(self, state: PipelineState) -> dict:
        started = time.perf_counter()
        try:
            result = self.searcher.search(
                state["request"].message, top_k=self.settings.retrieval_top_k
            )
        except Exception as error:
            state["trace"].add_error(type(error).__name__)
            return {"failed": True, "status": "error", "next": "failure"}
        state["trace"].set_retrieval(result.hits, (time.perf_counter() - started) * 1000)

        hits = self._within_cutoff(result.hits)
        if not hits:
            # A successful tool result is sufficient evidence on its own.
            if any(call.outcome.ok for call in state.get("tool_calls", [])):
                return {"hits": [], "failed": False, "next": "compose"}
            return {"hits": [], "escalation_reason": EscalationReason.MISSING_KNOWLEDGE,
                    "failed": False, "next": "escalate"}
        return {"hits": hits, "failed": False, "next": "compose"}

    def compose(self, state: PipelineState) -> dict:
        decision = state["decision"]
        hits = state.get("hits", [])
        calls = state.get("tool_calls", [])
        started = time.perf_counter()
        try:
            result = self.model.compose(
                message=state["request"].message,
                history=state["history"],
                hits=hits,
                tool_calls=calls,
            )
        except Exception as error:
            state["trace"].add_error(type(error).__name__)
            return {"failed": True, "status": "error", "next": "failure"}
        state["trace"].add_llm(result.usage, (time.perf_counter() - started) * 1000)

        generated = result.value
        cited = output_guard.validate_citations(
            generated.source_ids, {hit.article_id for hit in hits}
        )
        # Policy-only answers must cite evidence. A narrow policy answer may stand on
        # confirmed tool output alone, so it is exempt from the citation requirement.
        needs_citation = decision.action is Action.ANSWER_POLICY and not any(
            call.outcome.ok for call in calls
        )
        if not generated.supported or (needs_citation and not cited):
            # No verifiable evidence: hand off instead of inventing policy.
            return {"escalation_reason": EscalationReason.MISSING_KNOWLEDGE,
                    "failed": False, "next": "escalate"}

        wrote = any(call.outcome.ok and call.tool_name in WRITE_TOOL_NAMES for call in calls)
        return {
            "answer": generated.answer,
            "source_ids": cited,
            "status": "action_completed" if wrote else "answered",
            "next": "check_output",
        }

    # --- tools -------------------------------------------------------------
    def tool(self, state: PipelineState) -> dict:
        decision = state["decision"]
        call = self.executor.execute(
            name=decision.tool_name,
            arguments=decision.tool_arguments or {},
            session=state["session"],
            confirmed=state["request"].confirm_write,
        )
        state["trace"].add_tool(call)
        code = call.outcome.code

        if code == "ok":
            # Scenario A: attach policy guidance to the factual tool answer.
            nxt = "retrieve" if decision.needs_retrieval else "compose"
            return {"tool_calls": [call], "failed": False, "next": nxt}

        if code == "confirmation_required":
            subject = (decision.tool_arguments or {}).get("subject", "support request")
            return {
                "tool_calls": [call],
                "answer": CONFIRM_TEMPLATE.format(subject=subject[:80]),
                "status": "clarification_required",
                "source_ids": [],
                "next": "check_output",
            }

        if code == "not_found":
            return {
                "tool_calls": [call],
                "answer": NOT_FOUND_ANSWER,
                "status": "insufficient_context",
                "source_ids": [],
                "next": "check_output",
            }

        if code == "not_authorized":
            return {"tool_calls": [call], "status": "refused", "next": "refuse"}

        # invalid_arguments, unknown_tool, execution_failed: stop and hand off.
        return {
            "tool_calls": [call],
            "escalation_reason": EscalationReason.TOOL_FAILURE,
            "failed": False,
            "next": "escalate",
        }

    # --- escalation --------------------------------------------------------
    def _handoff(self, state: PipelineState, reason: EscalationReason) -> EscalationHandoff:
        decision = state.get("decision")
        message = state["request"].message
        history = state["history"]
        category = decision.category if decision else Category.SUPPORT_SCOPE
        turns = f" ({len(history)} earlier turns)" if history else ""
        return EscalationHandoff(
            reason=reason,
            category=category,
            summary=f"{category.value}: {message[:280]}{turns}"[:600],
            detected_issue=DETECTED_ISSUE[reason],
            evidence=[
                EvidenceRef(article_id=hit.article_id, title=hit.title, rank=rank,
                            distance=hit.distance)
                for rank, hit in enumerate(state.get("hits", []), 1)
            ],
            tools_attempted=[
                ToolAttempt(tool_name=call.tool_name, code=call.outcome.code)
                for call in state.get("tool_calls", [])
            ],
            trace_id=state["trace"].trace_id,
            created_at=datetime.now(timezone.utc),
        )

    def escalate(self, state: PipelineState) -> dict:
        reason = state.get("escalation_reason") or EscalationReason.REQUIRES_HUMAN_REVIEW
        handoff = self._handoff(state, reason)
        state["trace"].set_escalation(reason)

        # Escalation writes on the customer's behalf: the customer asked for this
        # outcome (explicitly or by hitting an unsupported path), so no extra
        # confirmation round is required. This is documented in DECISIONS.md.
        call = self.executor.execute(
            name="create_support_ticket",
            arguments={
                "subject": f"Escalation: {reason.value} ({handoff.category.value})"[:200],
                "message": json.dumps(handoff.model_dump(mode="json"), sort_keys=True,
                                      ensure_ascii=True)[:3900],
            },
            session=state["session"],
            confirmed=True,
        )
        state["trace"].add_tool(call)
        existing = list(state.get("tool_calls", []))

        if not call.outcome.ok:
            state["trace"].add_error("EscalationWriteFailed")
            return {
                "escalation": handoff,
                "tool_calls": existing + [call],
                "answer": (
                    "I could not record a handoff ticket, so a human has not been notified. "
                    "No other action was performed."
                ),
                "status": "error",
                "source_ids": [],
                "next": "failure",
            }

        ticket_id = (call.outcome.payload or {}).get("id")
        return {
            "escalation": handoff,
            "escalation_ticket_id": ticket_id,
            "tool_calls": existing + [call],
            "answer": ESCALATION_ANSWERS[reason].format(ticket=ticket_id),
            "status": "escalated",
            "source_ids": [],
            "next": "check_output",
        }

    # --- final gate --------------------------------------------------------
    def check_output(self, state: PipelineState) -> dict:
        calls = state.get("tool_calls", [])
        check = output_guard.validate_answer(
            state.get("answer", ""),
            customer_id=state["session"].customer_id,
            tool_calls=calls,
        )
        if check.violations:
            state["trace"].add_flags([violation.value for violation in check.violations])
            if check.action_claim_blocked:
                status = "unverified_action"
            elif output_guard.Violation.FOREIGN_RECORD in check.violations:
                status = "refused"
            else:
                status = "error"
            return {"answer": check.answer, "status": status, "source_ids": []}

        answer = check.answer
        source_ids = state.get("source_ids", [])
        if source_ids:
            titles = {hit.article_id: hit.title for hit in state.get("hits", [])}
            cited = ", ".join(f"{i} - {titles[i]}" for i in source_ids if i in titles)
            if cited:
                answer = f"{answer}\n\nSources: {cited}"
        return {"answer": answer}

    def failure(self, state: PipelineState) -> dict:
        return {
            "answer": state.get("answer") or ERROR_ANSWER,
            "status": "error",
            "source_ids": [],
        }
