"""Shared fixtures for pipeline tests.

Every harness owns a temporary seeded SQLite database, so tests never touch the
developer's data/support.sqlite3 and never share state.
"""

import tempfile
from pathlib import Path

from app.agent.pipeline import SupportPipeline
from app.config import Settings
from app.observability import MemoryTraceSink
from app.schemas.decision import AgentDecision
from app.schemas.pipeline import (
    ConversationTurn,
    PipelineRequest,
    SessionContext,
    ToolCallRecord,
    ToolOutcome,
)
from app.storage.database import Database
from app.storage.repository import SupportRepository
from app.storage.seed import seed_database
from app.tools.support import SupportTools
from evaluation.offline_retriever import LexicalRetriever
from evaluation.scripted import ScriptedSupportModel
from scripts.validate_data import validate_dataset


def decision(**overrides) -> AgentDecision:
    base = dict(
        category="payments", action="answer_policy", needs_retrieval=True,
        needs_tool=False, needs_clarification=False, needs_escalation=False,
    )
    base.update(overrides)
    return AgentDecision(**base)


def tool_call(name: str, code: str, payload: dict | None = None,
              arguments: dict | None = None) -> ToolCallRecord:
    outcome = ToolOutcome(
        tool_name=name, code=code,
        message="ok" if code == "ok" else "not ok",
        payload=payload if code == "ok" else None,
    )
    return ToolCallRecord(tool_name=name, arguments=arguments or {}, outcome=outcome,
                          duration_ms=0.5)


class PipelineHarness:
    def __init__(self, decisions=(), answers=(), *, searcher=None,
                 repository_factory=SupportRepository, **setting_overrides):
        self._folder = tempfile.TemporaryDirectory()
        self.dataset = validate_dataset()
        self.database = Database(Path(self._folder.name) / "test.sqlite3")
        self.database.initialize()
        seed_database(self.database, self.dataset)
        self.repository = repository_factory(self.database)
        self.model = ScriptedSupportModel(list(decisions), list(answers))
        self.searcher = searcher or LexicalRetriever(self.dataset.knowledge_base)
        self.sink = MemoryTraceSink()
        self.settings = Settings(
            _env_file=None, database_path=self.database.path, **setting_overrides
        )
        self.tools = SupportTools(self.repository)
        self.pipeline = SupportPipeline(
            model=self.model, searcher=self.searcher, tools=self.tools,
            settings=self.settings, sink=self.sink,
        )
        self.history: list[ConversationTurn] = []

    def run(self, message, *, customer_id="usr_001", confirm_write=False, history=None):
        response = self.pipeline.run(
            PipelineRequest(
                message=message,
                history=list(self.history if history is None else history),
                confirm_write=confirm_write,
            ),
            SessionContext(customer_id=customer_id),
        )
        self.history = [
            *self.history,
            ConversationTurn(role="customer", content=message),
            ConversationTurn(role="agent", content=response.answer),
        ]
        return response

    @property
    def trace(self):
        return self.sink.records[-1]

    def close(self) -> None:
        self._folder.cleanup()


def harness(testcase, *args, **kwargs) -> PipelineHarness:
    """Build a harness and register cleanup on the given unittest case."""
    instance = PipelineHarness(*args, **kwargs)
    testcase.addCleanup(instance.close)
    return instance
