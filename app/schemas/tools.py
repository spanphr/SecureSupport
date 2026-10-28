"""Validated arguments and results for deterministic, owner-scoped tools."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.domain import Complaint, Record

ComplaintStatus = Literal['open', 'investigating', 'resolved', 'closed']


class ToolContext(Record):
    # Supplied by the calling application, never extracted from a chat message.
    # This is an identity contract, not authentication.
    user_id: str = Field(min_length=1, max_length=100)


class NoArguments(Record):
    """Marker for tools that take no model-supplied arguments."""


class RecordLookup(Record):
    record_id: str = Field(min_length=1, max_length=100)


class AccountStatus(Record):
    """Minimal account view: no name, email, or other profile data."""

    user_id: str = Field(min_length=1, max_length=100)
    account_state: Literal["active"] = "active"
    mfa_enabled: bool


class CreateTicket(Record):
    subject: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=2000)


class CreateComplaint(Record):
    ticket_id: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=4000)


class UpdateComplaint(Record):
    complaint_id: str = Field(min_length=1, max_length=100)
    expected_version: int = Field(strict=True, ge=1)
    description: str | None = Field(default=None, min_length=1, max_length=4000)
    status: ComplaintStatus | None = None

    @model_validator(mode='after')
    def require_change(self):
        if self.description is None and self.status is None:
            raise ValueError('Provide a description or status update.')
        return self


class DeleteComplaint(Record):
    complaint_id: str = Field(min_length=1, max_length=100)
    expected_version: int = Field(strict=True, ge=1)


class ComplaintRecord(Complaint):
    updated_at: datetime
    version: int = Field(strict=True, ge=1)


class DeletedComplaint(Record):
    complaint_id: str
    deleted: Literal[True] = True
