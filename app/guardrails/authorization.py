"""Least-privilege policy for tool calls.

Identity flows one way: SessionContext (trusted) -> ToolContext -> SQL owner scope.
Nothing in the decision object can widen that. Record ownership itself is enforced
in app.storage.repository, where every query binds the owner id.
"""

from dataclasses import dataclass
from typing import Literal

from app.schemas.pipeline import SessionContext
from app.schemas.tools import ToolContext

ToolKind = Literal["read", "write"]


@dataclass(frozen=True)
class Denial:
    code: Literal["not_authorized", "confirmation_required"]
    message: str


def tool_context(session: SessionContext) -> ToolContext:
    """The only bridge from session identity to tool identity."""
    return ToolContext(user_id=session.customer_id)


def authorize_call(
    *, kind: ToolKind, requires_confirmation: bool, confirmation_supplied: bool
) -> Denial | None:
    """Return a denial, or None when the call may proceed."""
    if kind == "write" and requires_confirmation and not confirmation_supplied:
        return Denial(
            "confirmation_required",
            "This changes stored data and needs explicit confirmation first.",
        )
    return None
