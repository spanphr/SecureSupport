"""Validated synthetic support records; monetary amounts use integer cents."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class User(Record):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.invalid$")
    mfa_enabled: bool


class Payment(Record):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    amount_cents: int = Field(strict=True, gt=0)
    currency: Literal["AUD"] = "AUD"
    status: Literal["pending", "completed", "failed", "refunded"]
    created_at: datetime


class Ticket(Record):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    message: str = Field(min_length=1)
    status: Literal["open", "in_progress", "resolved"]
    created_at: datetime


class KnowledgeArticle(Record):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    tags: list[str]


class Complaint(Record):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    ticket_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: Literal["open", "investigating", "resolved", "closed"]
    created_at: datetime


class FailureCase(Record):
    id: str = Field(min_length=1)
    category: Literal["prompt_injection", "mfa_bypass", "data_leakage", "unsupported_advice", "false_action_claim", "missing_fields"]
    prompt: str = Field(min_length=1)
    expected_behavior: str = Field(min_length=1)


class Dataset(Record):
    users: list[User]
    payments: list[Payment]
    tickets: list[Ticket]
    knowledge_base: list[KnowledgeArticle]
    complaints: list[Complaint]
    failure_cases: list[FailureCase]
