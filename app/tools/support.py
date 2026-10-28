"""Deterministic tool boundary.

No LLM calls, no prompt interpretation, no approvals logic. Every operation is
owner-scoped by ToolContext and returns a validated record or a safe domain error.
"""

from app.schemas.domain import Payment, Ticket
from app.schemas.tools import (
    AccountStatus,
    CreateComplaint,
    CreateTicket,
    DeleteComplaint,
    NoArguments,
    RecordLookup,
    ToolContext,
    UpdateComplaint,
)
from app.storage.repository import SupportRepository


class SupportTools:
    def __init__(self, repository: SupportRepository):
        self.repository = repository

    # --- account -----------------------------------------------------------
    def get_account_status(self, context: ToolContext, request: NoArguments) -> AccountStatus:
        return self.repository.get_account(context.user_id)

    # --- payments and tickets (read-only) ----------------------------------
    def get_payment_status(self, context: ToolContext, request: RecordLookup) -> Payment:
        return self.repository.get_payment(context.user_id, request.record_id)

    def get_ticket_status(self, context: ToolContext, request: RecordLookup) -> Ticket:
        return self.repository.get_ticket(context.user_id, request.record_id)

    def get_payment(self, context: ToolContext, request: RecordLookup) -> Payment:
        """Alias retained for the earlier developer CLI."""
        return self.get_payment_status(context, request)

    def get_ticket(self, context: ToolContext, request: RecordLookup) -> Ticket:
        """Alias retained for the earlier developer CLI."""
        return self.get_ticket_status(context, request)

    # --- tickets (state-changing) ------------------------------------------
    def create_support_ticket(self, context: ToolContext, request: CreateTicket) -> Ticket:
        return self.repository.create_ticket(context.user_id, request)

    # --- complaints (existing CRUD) ----------------------------------------
    def get_complaint(self, context: ToolContext, request: RecordLookup):
        return self.repository.get_complaint(context.user_id, request.record_id)

    def list_complaints(self, context: ToolContext):
        return self.repository.list_complaints(context.user_id)

    def create_complaint(self, context: ToolContext, request: CreateComplaint):
        return self.repository.create_complaint(context.user_id, request)

    def update_complaint(self, context: ToolContext, request: UpdateComplaint):
        return self.repository.update_complaint(context.user_id, request)

    def delete_complaint(self, context: ToolContext, request: DeleteComplaint):
        return self.repository.delete_complaint(context.user_id, request)
