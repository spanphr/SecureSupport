"""Scripted demo mode for the browser UI.

Replays the four documented scenarios through the real pipeline: real lexical
retrieval, real SQLite tools, real guardrails, real escalation tickets. Only the
model is scripted, so the demo is free, deterministic, and safe to run in CI.

Unknown messages are refused with a pointer to the preset questions rather than
being answered by nothing: a scripted model has no opinion about anything else.
"""

import re
import tempfile
from pathlib import Path

from app.agent.pipeline import SupportPipeline
from app.config import Settings
from app.demo_scenarios import SCENARIOS, Scenario
from app.observability import TraceSink
from app.storage.database import Database
from app.storage.repository import SupportRepository
from app.storage.seed import seed_database
from app.tools.support import SupportTools
from evaluation.offline_retriever import LexicalRetriever
from evaluation.scripted import ScriptedSupportModel
from scripts.validate_data import validate_dataset

_NON_WORD = re.compile(r"[^a-z0-9\s]+")


def normalize(text: str) -> str:
    """Compare messages on words alone: case, punctuation, and spacing do not matter."""
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


class ScriptedDemo:
    """Owns a disposable seeded database and builds one pipeline per scripted turn."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._folder = tempfile.TemporaryDirectory(prefix="securesupport-demo-")
        self.dataset = validate_dataset()
        self.database = Database(Path(self._folder.name) / "demo.sqlite3")
        self.database.initialize()
        seed_database(self.database, self.dataset)
        self.searcher = LexicalRetriever(self.dataset.knowledge_base)
        self._by_message = {normalize(s.message): s for s in SCENARIOS.values()}

    def match(self, message: str) -> Scenario | None:
        return self._by_message.get(normalize(message))

    def pipeline(self, scenario: Scenario, sink: TraceSink) -> SupportPipeline:
        decisions = [scenario.decision] if scenario.decision else []
        answers = [scenario.answer] if scenario.answer else []
        return SupportPipeline(
            model=ScriptedSupportModel(decisions, answers),
            searcher=self.searcher,
            tools=SupportTools(SupportRepository(self.database)),
            settings=self.settings,
            sink=sink,
        )

    def close(self) -> None:
        self._folder.cleanup()
