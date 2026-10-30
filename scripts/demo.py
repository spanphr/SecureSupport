"""Runnable end-to-end demonstration of the support pipeline.

Offline by default: real SQLite tools, real retrieval ranking (deterministic lexical),
real guardrails and escalation, with a scripted decision model so the run is
reproducible and free. `--live` uses the configured OpenAI model and Chroma index,
which spends tokens on your key.

    python -m scripts.demo                 # all four scenarios, offline
    python -m scripts.demo --scenario A    # one scenario
    python -m scripts.demo --live          # real model + embeddings (paid)
"""

import argparse
import tempfile
from pathlib import Path

from app.agent.factory import build_pipeline
from app.agent.pipeline import SupportPipeline
from app.config import Settings
from app.demo_scenarios import SCENARIOS, Scenario
from app.observability import MemoryTraceSink
from app.schemas.pipeline import PipelineRequest, SessionContext
from app.storage.database import Database
from app.storage.repository import SupportRepository
from app.storage.seed import seed_database
from app.tools.support import SupportTools
from evaluation.offline_retriever import LexicalRetriever
from evaluation.scripted import ScriptedSupportModel
from scripts.validate_data import validate_dataset


def _offline(scenario: Scenario, dataset, database) -> tuple:
    decisions = [scenario.decision] if scenario.decision else []
    answers = [scenario.answer] if scenario.answer else []
    sink = MemoryTraceSink()
    pipeline = SupportPipeline(
        model=ScriptedSupportModel(decisions, answers),
        searcher=LexicalRetriever(dataset.knowledge_base),
        tools=SupportTools(SupportRepository(database)),
        settings=Settings(_env_file=None, database_path=database.path),
        sink=sink,
    )
    return pipeline, sink


def _report(scenario: Scenario, response, trace) -> None:
    print("=" * 78)
    print(f"Scenario {scenario.key}: {scenario.title}")
    print(f"Customer ({scenario.customer_id}): {scenario.message}")
    print("-" * 78)
    print(f"status          : {response.status}")
    print(f"category/action : {response.category} / {response.action}"
          if response.category else "category/action : (not decided)")
    print(f"sources         : {response.source_ids or '-'}")
    print(f"tool calls      : "
          f"{[(c.tool_name, c.outcome.code) for c in response.tool_calls] or '-'}")
    print(f"action_performed: {response.action_performed}")
    if response.escalation:
        print(f"escalation      : {response.escalation.reason.value} "
              f"-> {response.escalation.detected_issue}")
        print(f"handoff ticket  : {response.escalation_ticket_id or 'not created'}")
    print(f"trace           : {response.trace_id} "
          f"({response.total_latency_ms:.1f} ms, {trace.llm_calls} LLM calls, "
          f"{trace.input_tokens}+{trace.output_tokens} tokens, "
          f"cost {trace.estimated_cost_usd if trace.estimated_cost_usd is not None else 'unpriced'})")
    print("answer:")
    for line in response.answer.splitlines():
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SecureSupport demo scenarios.")
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--live", action="store_true",
                        help="Use the configured OpenAI model and Chroma index (spends tokens).")
    args = parser.parse_args()

    keys = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    dataset = validate_dataset()

    if args.live:
        print("LIVE MODE: this sends messages to the configured chat model and retrieval "
              "index and incurs API usage on your key.")
        sink = MemoryTraceSink()
        pipeline = build_pipeline(sink=sink)
        for key in keys:
            scenario = SCENARIOS[key]
            response = pipeline.run(PipelineRequest(message=scenario.message),
                                    SessionContext(customer_id=scenario.customer_id))
            _report(scenario, response, sink.records[-1])
        return

    print("OFFLINE MODE: scripted decisions, real tools, real guardrails, seeded temp database.")
    with tempfile.TemporaryDirectory() as folder:
        database = Database(Path(folder) / "demo.sqlite3")
        database.initialize()
        seed_database(database, dataset)
        for key in keys:
            scenario = SCENARIOS[key]
            pipeline, sink = _offline(scenario, dataset, database)
            response = pipeline.run(PipelineRequest(message=scenario.message),
                                    SessionContext(customer_id=scenario.customer_id))
            _report(scenario, response, sink.records[-1])


if __name__ == "__main__":
    main()
