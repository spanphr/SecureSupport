"""Output validation.

The model may not assert an action completed without a confirming tool result, may
not cite articles it was not given, and may not surface identifiers belonging to
another customer. Blocked answers are replaced, never merely annotated.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

from app.schemas.pipeline import ToolCallRecord

SAFE_FALLBACK_ANSWER = (
    "I could not verify that answer against the available support policies. "
    "No account action was performed."
)
UNVERIFIED_ACTION_ANSWER = (
    "I can't confirm that any action was completed. This assistant has not issued a "
    "refund, cancelled anything, or changed an account."
)
WRITE_TOOL_NAMES = frozenset({"create_support_ticket"})

_COMPLETION_CLAIMS = (
    re.compile(
        r"\b(?:i|we)(?:'ve|\s+have)\s+(?:now\s+)?(?:refunded|processed|cancelled|canceled|"
        r"updated|created|opened|escalated|deleted|submitted|changed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:your|the)\s+(?:refund|payment|ticket|account|complaint|escalation|request)\s+"
        r"(?:has|have)\s+been\s+(?:processed|created|completed|issued|approved|updated|"
        r"cancelled|canceled|refunded|submitted|escalated)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsuccessfully\s+(?:refunded|processed|cancelled|canceled|updated|created|escalated|"
        r"deleted|submitted)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:refund|cancellation|update|ticket|escalation)\s+(?:is|was)\s+(?:now\s+)?"
        r"(?:complete|completed|processed|created|done)\b",
        re.IGNORECASE,
    ),
)

# Only identifiers this system issues. Deliberately excludes article ids (kb_*),
# which are public documentation, not customer records.
_RECORD_ID = re.compile(r"\b(?:usr|pay|tkt|cmp)_[A-Za-z0-9]{2,}\b")
_NEGATION = re.compile(r"\b(?:no|not|never|cannot|can't|haven't|hasn't|didn't|unable)\b", re.IGNORECASE)


class Violation(StrEnum):
    EMPTY_ANSWER = "empty_answer"
    FOREIGN_RECORD = "foreign_record"
    ACTION_CLAIM = "action_claim"


@dataclass(frozen=True)
class OutputCheck:
    answer: str
    violations: tuple[Violation, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.violations

    @property
    def action_claim_blocked(self) -> bool:
        return Violation.ACTION_CLAIM in self.violations


def _is_negated(text: str, start: int) -> bool:
    """'No refund has been processed' is a denial, not a completion claim."""
    return bool(_NEGATION.search(text[max(0, start - 20):start]))


def claims_completed_action(answer: str) -> bool:
    for pattern in _COMPLETION_CLAIMS:
        for match in pattern.finditer(answer):
            if not _is_negated(answer, match.start()):
                return True
    return False


def _confirmed_write(tool_calls: list[ToolCallRecord]) -> bool:
    return any(call.outcome.ok and call.tool_name in WRITE_TOOL_NAMES for call in tool_calls)


def _owned_record_ids(tool_calls: list[ToolCallRecord], customer_id: str) -> set[str]:
    """Identifiers a successful tool returned for this customer."""
    owned: set[str] = set()
    for call in tool_calls:
        payload = call.outcome.payload
        if not call.outcome.ok or payload is None:
            continue
        # Only trust a payload that names the session customer as its owner.
        if payload.get("user_id") not in (None, customer_id):
            continue
        if isinstance(payload.get("id"), str):
            owned.add(payload["id"])
        if isinstance(payload.get("user_id"), str):
            owned.add(payload["user_id"])
        if isinstance(payload.get("ticket_id"), str):
            owned.add(payload["ticket_id"])
    return owned


def validate_answer(
    answer: str, *, customer_id: str, tool_calls: list[ToolCallRecord]
) -> OutputCheck:
    text = answer.strip()
    if not text:
        return OutputCheck(SAFE_FALLBACK_ANSWER, (Violation.EMPTY_ANSWER,))

    allowed = _owned_record_ids(tool_calls, customer_id)
    if any(identifier not in allowed for identifier in _RECORD_ID.findall(text)):
        return OutputCheck(SAFE_FALLBACK_ANSWER, (Violation.FOREIGN_RECORD,))

    if claims_completed_action(text) and not _confirmed_write(tool_calls):
        return OutputCheck(UNVERIFIED_ACTION_ANSWER, (Violation.ACTION_CLAIM,))

    return OutputCheck(text)


def validate_citations(source_ids: list[str], retrieved_ids: set[str]) -> list[str]:
    """Keep only citations the model was actually shown, deduplicated and ordered."""
    return list(dict.fromkeys(identifier for identifier in source_ids if identifier in retrieved_ids))
