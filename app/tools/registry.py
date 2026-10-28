"""Tool registry: allowlist, argument validation, authorization, structured outcomes.

Design rules taken from the agentic-workflow literature applied here:
- Tools are the only way to touch state, and only listed tools can be called.
- Arguments are validated by a Pydantic model before any handler runs.
- Failures are values (ToolOutcome), so a bad tool cannot crash the request.
- Read and write tools are disjoint sets; writes declare whether they need confirmation.
"""

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from app.guardrails.authorization import ToolKind, authorize_call, tool_context
from app.schemas.domain import Record
from app.schemas.pipeline import SessionContext, ToolCallRecord, ToolOutcome, ToolOutcomeCode
from app.schemas.tools import NoArguments, RecordLookup, CreateTicket
from app.storage.errors import RecordNotFound, StorageError
from app.tools.support import SupportTools

Handler = Callable[[SupportTools, object, object], BaseModel]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    arguments_model: type[Record]
    kind: ToolKind
    requires_confirmation: bool
    handler: Handler


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="get_account_status",
            description="Read the authenticated customer's own account status, including MFA state.",
            arguments_model=NoArguments,
            kind="read",
            requires_confirmation=False,
            handler=lambda tools, context, args: tools.get_account_status(context, args),
        ),
        ToolSpec(
            name="get_payment_status",
            description="Read one payment belonging to the authenticated customer.",
            arguments_model=RecordLookup,
            kind="read",
            requires_confirmation=False,
            handler=lambda tools, context, args: tools.get_payment_status(context, args),
        ),
        ToolSpec(
            name="get_ticket_status",
            description="Read one support ticket belonging to the authenticated customer.",
            arguments_model=RecordLookup,
            kind="read",
            requires_confirmation=False,
            handler=lambda tools, context, args: tools.get_ticket_status(context, args),
        ),
        ToolSpec(
            name="create_support_ticket",
            description=(
                "Open a new support ticket for the authenticated customer. "
                "Requires explicit confirmation because it writes a record."
            ),
            arguments_model=CreateTicket,
            kind="write",
            requires_confirmation=True,
            handler=lambda tools, context, args: tools.create_support_ticket(context, args),
        ),
    )
}

READ_TOOL_NAMES = frozenset(name for name, spec in TOOL_SPECS.items() if spec.kind == "read")
WRITE_TOOL_NAMES = frozenset(name for name, spec in TOOL_SPECS.items() if spec.kind == "write")
assert not READ_TOOL_NAMES & WRITE_TOOL_NAMES, "A tool cannot be both read and write."


def catalog() -> list[dict[str, str]]:
    """Descriptions passed to the model. The model never sees unlisted tools."""
    return [
        {"name": spec.name, "description": spec.description, "kind": spec.kind,
         "confirmed": str(spec.requires_confirmation)}
        for spec in TOOL_SPECS.values()
    ]


def outcome(name: str, code: ToolOutcomeCode, message: str,
            payload: dict | None = None) -> ToolOutcome:
    return ToolOutcome(tool_name=name, code=code, message=message, payload=payload)


class ToolExecutor:
    """Executes one allowlisted tool call. Never raises for expected failures."""

    def __init__(self, tools: SupportTools):
        self.tools = tools

    def execute(
        self,
        *,
        name: str,
        arguments: dict[str, str],
        session: SessionContext,
        confirmed: bool = False,
    ) -> ToolCallRecord:
        started = time.perf_counter()
        result = self._run(name=name, arguments=arguments, session=session, confirmed=confirmed)
        return ToolCallRecord(
            tool_name=name,
            arguments=dict(arguments),
            outcome=result,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def _run(
        self, *, name: str, arguments: dict[str, str], session: SessionContext, confirmed: bool
    ) -> ToolOutcome:
        spec = TOOL_SPECS.get(name)
        if spec is None:
            # Unknown or restricted operations (refunds, deletions) land here.
            return outcome(name or "unknown", "unknown_tool",
                           "That operation is not available to this assistant.")

        try:
            parsed = spec.arguments_model.model_validate(arguments)
        except ValidationError:
            return outcome(name, "invalid_arguments", "Tool arguments were rejected.")

        denial = authorize_call(
            kind=spec.kind,
            requires_confirmation=spec.requires_confirmation,
            confirmation_supplied=confirmed,
        )
        if denial is not None:
            return outcome(name, denial.code, denial.message)

        try:
            record = spec.handler(self.tools, tool_context(session), parsed)
            return outcome(name, "ok", "Operation completed.",
                           payload=record.model_dump(mode="json"))
        except RecordNotFound:
            # Missing and another customer's record are the same answer.
            return outcome(name, "not_found", "No matching record was found for this account.")
        except (StorageError, sqlite3.Error, OSError):
            return outcome(name, "execution_failed",
                           "The operation could not be completed; no change was confirmed.")
        except Exception:
            # Defensive: a handler bug must not leak diagnostics or crash the request.
            return outcome(name, "execution_failed",
                           "The operation could not be completed; no change was confirmed.")
