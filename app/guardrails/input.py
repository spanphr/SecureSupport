"""Input validation.

Heuristics only *flag* intent; they never grant permission. Structural controls
(tool allowlist, owner-scoped SQL, output checks) are what actually enforce policy.
Retrieved documentation is data, never instruction - see app.guardrails.output.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

MAX_MESSAGE_CHARS = 4000

_INJECTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instructions?|rules?|prompts?)"
    r"|disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above|your)\s+(?:instructions?|rules?)"
    r"|you\s+are\s+now\b"
    r"|reveal\s+(?:your\s+)?(?:system\s+prompt|instructions|hidden\s+instructions)"
    r"|\bsystem\s+prompt\b"
    r"|\bdeveloper\s+message\b"
    r"|\bjailbreak\b",
    re.IGNORECASE,
)

_SECRET_REQUEST = re.compile(
    r"\b(?:api\s*key|secret\s*key|access\s*token|credentials?|environment\s+variables?)\b"
    r"|\binternal\s+(?:instructions|prompt|secrets?)\b",
    re.IGNORECASE,
)

_CROSS_CUSTOMER = re.compile(
    r"\b(?:another|other|different)\s+(?:customer|user|client|account|person)s?\b"
    r"|\bsomeone\s+else(?:'s)?\b"
    r"|\bother\s+(?:people|users|customers)(?:'s)?\b",
    re.IGNORECASE,
)

# Operations with no tool in this system. Detection blocks completion claims; the
# allowlist already makes them impossible to execute.
_RESTRICTED = re.compile(
    r"\b(?:issue|process|approve|execute)\s+(?:a\s+|the\s+|my\s+)?refund\b"
    r"|\brefund\s+(?:me|pay_\w+)\b"
    r"|\bcancel\s+(?:my\s+|the\s+)?(?:account|subscription|order)\b"
    r"|\b(?:disable|turn\s+off|bypass|remove)\s+(?:my\s+)?(?:mfa|2fa|two[- ]factor|verification)\b"
    r"|\bdelete\s+(?:my\s+|the\s+)?account\b",
    re.IGNORECASE,
)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InputCode(StrEnum):
    OK = "ok"
    INVALID = "invalid"
    INJECTION_SIGNAL = "injection_signal"
    SECRET_REQUEST = "secret_request"
    CROSS_CUSTOMER = "cross_customer"
    RESTRICTED_OPERATION = "restricted_operation"


@dataclass(frozen=True)
class InputAssessment:
    """`ok=False` means refuse before retrieval or tool execution."""

    ok: bool
    code: InputCode
    flags: tuple[str, ...] = ()
    reason: str = ""


def assess(message: str) -> InputAssessment:
    text = message.strip()
    if not text:
        return InputAssessment(False, InputCode.INVALID, reason="Message is empty.")
    if len(text) > MAX_MESSAGE_CHARS:
        return InputAssessment(False, InputCode.INVALID, reason="Message exceeds 4000 characters.")
    if _CONTROL.search(text):
        return InputAssessment(False, InputCode.INVALID, reason="Message contains control characters.")

    flags: list[str] = []
    if _INJECTION.search(text):
        return InputAssessment(
            False, InputCode.INJECTION_SIGNAL, ("injection",),
            "Message tries to override assistant instructions.",
        )
    if _SECRET_REQUEST.search(text):
        return InputAssessment(
            False, InputCode.SECRET_REQUEST, ("secret_request",),
            "Message requests credentials or internal instructions.",
        )
    if _CROSS_CUSTOMER.search(text):
        # Authorization is decided by code, never by which article ranked first.
        return InputAssessment(
            False, InputCode.CROSS_CUSTOMER, ("cross_customer",),
            "Message requests another customer's records.",
        )
    if _RESTRICTED.search(text):
        flags.append("restricted_operation")
    return InputAssessment(True, InputCode.OK, tuple(flags))
