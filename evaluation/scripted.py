"""Deterministic model doubles.

Used by unit tests and the offline evaluation harness. These prove orchestration and
guardrail behaviour, not model quality; live model quality is a separate, opt-in run.
"""

from collections import deque

from app.agent.model import LLMResult
from app.schemas.decision import AgentDecision
from app.schemas.pipeline import ConversationTurn, GroundedAnswer, LLMUsage, ToolCallRecord
from app.schemas.retrieval import RetrievalHit, RetrievalResult

DEFAULT_USAGE = LLMUsage(input_tokens=60, output_tokens=20)


class ScriptedSupportModel:
    """Replays a fixed decision/composition script and records what it received."""

    def __init__(self, decisions: list[AgentDecision], answers: list[GroundedAnswer] | None = None,
                 *, usage: LLMUsage = DEFAULT_USAGE):
        self._decisions = deque(decisions)
        self._answers = deque(answers or [])
        self.usage = usage
        self.calls: list[dict] = []
        self.decision_error: Exception | None = None
        self.compose_error: Exception | None = None

    def decide(self, *, message: str, history: list[ConversationTurn],
               tools: list[dict[str, str]]) -> LLMResult[AgentDecision]:
        self.calls.append({"kind": "decide", "message": message, "history": list(history),
                           "tools": tools})
        if self.decision_error is not None:
            raise self.decision_error
        if not self._decisions:
            raise AssertionError("No scripted decision remains.")
        return LLMResult(value=self._decisions.popleft(), usage=self.usage)

    def compose(self, *, message: str, history: list[ConversationTurn],
                hits: list[RetrievalHit],
                tool_calls: list[ToolCallRecord]) -> LLMResult[GroundedAnswer]:
        self.calls.append({"kind": "compose", "message": message, "hits": list(hits),
                           "tool_calls": list(tool_calls)})
        if self.compose_error is not None:
            raise self.compose_error
        if not self._answers:
            raise AssertionError("No scripted answer remains.")
        return LLMResult(value=self._answers.popleft(), usage=self.usage)


class FixedSearcher:
    """Returns a fixed ranking, ignoring the query."""

    def __init__(self, hits: list[RetrievalHit]):
        self.hits = list(hits)
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        self.queries.append((query, top_k))
        return RetrievalResult(query=query, hits=self.hits[:top_k])


class FailingSearcher:
    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        raise RuntimeError("retrieval-backend-unavailable")


def hit(article_id: str, title: str = "", content: str = "", distance: float = 0.1) -> RetrievalHit:
    return RetrievalHit(article_id=article_id, title=title or article_id,
                        content=content or f"{article_id} content", distance=distance)
