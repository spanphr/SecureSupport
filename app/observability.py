"""Structured per-request traces.

Every request emits one record with its trace id, decision, retrieval ranking,
tool outcomes, latencies, token usage, configured-price cost, escalation reason,
and error class names. Message text, answers, and record payloads are never
written to the trace: only a fingerprint and character count.
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Protocol

from app.schemas.domain import Record
from app.schemas.pipeline import EvidenceRef, EscalationReason, LLMUsage, ToolAttempt, ToolCallRecord
from app.schemas.retrieval import RetrievalHit

LOGGER = logging.getLogger("securesupport.pipeline")
_PER_MILLION = Decimal(1_000_000)
_COST_QUANTUM = Decimal("0.000001")


class TraceRecord(Record):
    trace_id: str
    started_at: datetime
    status: str | None = None
    category: str | None = None
    action: str | None = None
    retrieval_hit: list[EvidenceRef] = []
    retrieval_latency_ms: float | None = None
    llm_latency_ms: float | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    tool_calls: list[ToolAttempt] = []
    escalation_reason: str | None = None
    errors: list[str] = []
    flags: list[str] = []
    message_fingerprint: str
    message_chars: int
    total_latency_ms: float = 0.0


class TraceSink(Protocol):
    def emit(self, record: TraceRecord) -> None: ...


class NullTraceSink:
    def emit(self, record: TraceRecord) -> None:  # pragma: no cover - deliberate no-op
        return None


class MemoryTraceSink:
    """Test and demo sink; keeps emitted records in order."""

    def __init__(self):
        self.records: list[TraceRecord] = []

    def emit(self, record: TraceRecord) -> None:
        self.records.append(record)


class JsonLogTraceSink:
    """One JSON object per line on the `securesupport.pipeline` logger."""

    def emit(self, record: TraceRecord) -> None:
        LOGGER.info(json.dumps(record.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))


def fingerprint(text: str) -> str:
    """Stable correlation handle that does not reveal the message."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def estimate_cost(usage: LLMUsage, input_price: Decimal | None,
                  output_price: Decimal | None) -> Decimal | None:
    """Cost is unknown unless both prices are explicitly configured."""
    if input_price is None or output_price is None:
        return None
    total = (Decimal(usage.input_tokens) * input_price
             + Decimal(usage.output_tokens) * output_price) / _PER_MILLION
    return total.quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)


class TraceBuilder:
    """Mutable accumulator used while a request runs; frozen into a record at the end."""

    def __init__(self, message: str, *, input_price: Decimal | None = None,
                 output_price: Decimal | None = None):
        self._started_wall = datetime.now(timezone.utc)
        self._started = time.perf_counter()
        self._input_price = input_price
        self._output_price = output_price
        self.trace_id = uuid.uuid4().hex
        self.status: str | None = None
        self.category: str | None = None
        self.action: str | None = None
        self.retrieval_hit: list[EvidenceRef] = []
        self.retrieval_latency_ms: float | None = None
        self.llm_latency_ms: float = 0.0
        self.llm_calls = 0
        self.usage = LLMUsage()
        self.tool_calls: list[ToolAttempt] = []
        self.escalation_reason: str | None = None
        self.errors: list[str] = []
        self.flags: list[str] = []
        self._message_fingerprint = fingerprint(message)
        self._message_chars = len(message)

    def set_decision(self, category: str, action: str) -> None:
        self.category = category
        self.action = action

    def set_retrieval(self, hits: list[RetrievalHit], latency_ms: float) -> None:
        self.retrieval_hit = [
            EvidenceRef(article_id=hit.article_id, title=hit.title, rank=rank, distance=hit.distance)
            for rank, hit in enumerate(hits, 1)
        ]
        self.retrieval_latency_ms = round(latency_ms, 3)

    def add_llm(self, usage: LLMUsage, latency_ms: float) -> None:
        self.llm_calls += 1
        self.llm_latency_ms += latency_ms
        self.usage = LLMUsage(
            input_tokens=self.usage.input_tokens + usage.input_tokens,
            output_tokens=self.usage.output_tokens + usage.output_tokens,
        )

    def add_tool(self, call: ToolCallRecord) -> None:
        self.tool_calls.append(ToolAttempt(tool_name=call.tool_name, code=call.outcome.code))

    def add_error(self, error: str) -> None:
        # Error class names only: messages can contain customer data or credentials.
        self.errors.append(error)

    def add_flags(self, flags: tuple[str, ...] | list[str]) -> None:
        for flag in flags:
            if flag not in self.flags:
                self.flags.append(flag)

    def set_escalation(self, reason: EscalationReason) -> None:
        self.escalation_reason = reason.value

    def finish(self, status: str) -> TraceRecord:
        cost = estimate_cost(self.usage, self._input_price, self._output_price)
        return TraceRecord(
            trace_id=self.trace_id,
            started_at=self._started_wall,
            status=status,
            category=self.category,
            action=self.action,
            retrieval_hit=self.retrieval_hit,
            retrieval_latency_ms=self.retrieval_latency_ms,
            llm_latency_ms=round(self.llm_latency_ms, 3) if self.llm_calls else None,
            llm_calls=self.llm_calls,
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            estimated_cost_usd=None if cost is None else float(cost),
            tool_calls=self.tool_calls,
            escalation_reason=self.escalation_reason,
            errors=self.errors,
            flags=self.flags,
            message_fingerprint=self._message_fingerprint,
            message_chars=self._message_chars,
            total_latency_ms=round((time.perf_counter() - self._started) * 1000, 3),
        )
