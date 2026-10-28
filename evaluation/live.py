"""Live, opt-in evaluation against the real model and embedding index.

This makes paid API calls. It is the only module in the repository that does so
outside the CLI entry points, and it refuses to run without `--confirm-live`.

Scoring is pure and unit-tested offline in tests/test_live_eval.py; the network
path is never exercised by the test suite.
"""

import argparse
import json
import statistics
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable

from pydantic import Field

from app.agent.model import OpenAISupportModel
from app.agent.pipeline import SupportPipeline
from app.config import Settings
from app.guardrails.output import claims_completed_action
from app.observability import MemoryTraceSink, TraceRecord
from app.schemas.domain import Dataset, Record
from app.schemas.pipeline import PipelineRequest, PipelineResponse, SessionContext
from app.storage.database import Database
from app.storage.repository import SupportRepository
from app.storage.seed import seed_database
from app.tools.support import SupportTools
from evaluation.pipeline import foreign_record_ids
from scripts.validate_data import validate_dataset

CASES_PATH = Path(__file__).resolve().parent / "live_cases.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
OFFLINE_REPORT = REPORTS_DIR / "pipeline_offline.json"
LIVE_REPORT = REPORTS_DIR / "pipeline_live.json"


class LiveCase(Record):
    id: str
    scenario: str
    customer_id: str
    message: str = Field(min_length=1, max_length=4000)
    allowed_statuses: list[str] = Field(min_length=1)
    required_source: list[str] = Field(default_factory=list)
    expected_tool: str | None = None
    must_not_contain: list[str] = Field(default_factory=list)


def load_cases(path: Path = CASES_PATH) -> list[LiveCase]:
    cases = [LiveCase.model_validate(item) for item in json.loads(path.read_text())]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate live case ids.")
    return cases


def score_case(case: LiveCase, response: PipelineResponse, trace: TraceRecord,
               dataset: Dataset) -> dict:
    foreign = foreign_record_ids(dataset, case.customer_id)
    retrieved = [item.article_id for item in trace.retrieval_hit]
    leaked_payload = any(
        call.outcome.ok
        and call.outcome.payload
        and call.outcome.payload.get("user_id") not in (None, case.customer_id)
        for call in response.tool_calls
    )
    leaked_text = any(identifier in response.answer for identifier in foreign)
    unsupported = claims_completed_action(response.answer) and not any(
        call.outcome.ok and call.tool_name == "create_support_ticket"
        for call in response.tool_calls
    )
    tools = [call.tool_name for call in response.tool_calls]

    checks = {
        "status": response.status in case.allowed_statuses,
        "sources": (not case.required_source)
        or bool(set(case.required_source) & set(response.source_ids)),
        "tool": case.expected_tool is None or case.expected_tool in tools,
        "no_unsupported_claim": not unsupported,
        "no_authorization_leak": not (leaked_payload or leaked_text),
        "no_forbidden_text": all(text not in response.answer for text in case.must_not_contain),
        "citations": set(response.source_ids) <= set(retrieved),
    }
    return {
        "case_id": case.id,
        "scenario": case.scenario,
        "status": response.status,
        "allowed_statuses": case.allowed_statuses,
        "sources": response.source_ids,
        "retrieval_ids": retrieved,
        "required_source": case.required_source,
        "tools": tools,
        "expects_tool": case.expected_tool is not None,
        "escalation_ticket_id": response.escalation_ticket_id,
        "latency_ms": response.total_latency_ms,
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
        "estimated_cost_usd": trace.estimated_cost_usd,
        "answer": response.answer,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_cases(cases: list[LiveCase], dataset: Dataset,
              run_one: Callable[[LiveCase], tuple[PipelineResponse, TraceRecord]]) -> list[dict]:
    return [score_case(case, *run_one(case), dataset) for case in cases]


def _rate(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return sum(row["checks"][key] for row in rows) / len(rows)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("At least one live case is required.")
    tool_rows = [row for row in rows if row["expects_tool"]]
    retrieval_rows = [row for row in rows if row["required_source"]]
    latencies = [row["latency_ms"] for row in rows]
    costs = [row["estimated_cost_usd"] for row in rows]
    return {
        "cases": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "pass_rate": sum(row["passed"] for row in rows) / len(rows),
        "status_accuracy": _rate(rows, "status"),
        "source_accuracy": _rate(retrieval_rows, "sources"),
        "tool_accuracy": _rate(tool_rows, "tool"),
        "authorization_violations": sum(
            not row["checks"]["no_authorization_leak"] for row in rows),
        "unsupported_claims": sum(
            not row["checks"]["no_unsupported_claim"] for row in rows),
        "citation_violations": sum(not row["checks"]["citations"] for row in rows),
        "retrieval": {
            "cases": len(retrieval_rows),
            "hit_at_1": (
                sum(bool(row["retrieval_ids"])
                    and row["retrieval_ids"][0] in row["required_source"]
                    for row in retrieval_rows) / len(retrieval_rows)
                if retrieval_rows else None
            ),
            "hit_at_3": (
                sum(bool(set(row["retrieval_ids"][:3]) & set(row["required_source"]))
                    for row in retrieval_rows) / len(retrieval_rows)
                if retrieval_rows else None
            ),
        },
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3),
            "p50": round(statistics.median(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "tokens": {
            "input": sum(row["input_tokens"] for row in rows),
            "output": sum(row["output_tokens"] for row in rows),
        },
        "estimated_cost_usd": (
            None if any(cost is None for cost in costs) else round(sum(costs), 6)
        ),
    }


def compare_reports(offline: dict, live: dict) -> list[dict]:
    pipeline = offline["pipeline"]
    retrieval = offline["retrieval"]
    rows = [
        ("pipeline_cases", pipeline["cases"], live["cases"]),
        ("pass_rate", pipeline["cases_passed"] / pipeline["cases"], live["pass_rate"]),
        ("authorization_violations",
         pipeline["authorization_violations"], live["authorization_violations"]),
        ("unsupported_claims", pipeline["unsupported_claims"], live["unsupported_claims"]),
        ("citation_violations", pipeline["citation_violations"], live["citation_violations"]),
        ("retrieval_hit_at_1", retrieval["hit_at_1"], live["retrieval"]["hit_at_1"]),
        ("retrieval_hit_at_3", retrieval["hit_at_3"], live["retrieval"]["hit_at_3"]),
        ("mean_latency_ms", pipeline["mean_latency_ms"], live["latency_ms"]["mean"]),
    ]
    return [{"metric": name, "offline": left, "live": right} for name, left, right in rows]


def _print_comparison(rows: list[dict]) -> None:
    print("\nOffline (deterministic) vs live (real model):")
    for row in rows:
        print(f"  {row['metric']:<26} offline={row['offline']!s:<14} live={row['live']}")


def build_live_pipeline(settings: Settings, database: Database) -> tuple[SupportPipeline, MemoryTraceSink]:
    """Construct the real pipeline. Imports retrieval lazily so tests stay offline."""
    from app.retrieval.factory import build_retriever

    retriever = build_retriever()
    if not retriever.ready:
        raise SystemExit(
            "Knowledge index is missing or incomplete. Run `python -m scripts.index_knowledge` first."
        )
    sink = MemoryTraceSink()
    pipeline = SupportPipeline(
        model=OpenAISupportModel(settings),
        searcher=retriever,
        tools=SupportTools(SupportRepository(database)),
        settings=settings,
        sink=sink,
    )
    return pipeline, sink


def preflight(settings: Settings) -> None:
    if settings.environment == "production":
        raise SystemExit("Refusing to run the live evaluation against a production configuration.")
    key = settings.openai_api_key
    if key is None or not key.get_secret_value().strip():
        raise SystemExit(
            "No API key configured. Set SECURESUPPORT_OPENAI_API_KEY in your local .env first."
        )


def _price(raw: str | None, fallback: Decimal | None) -> Decimal | None:
    if raw is None:
        return fallback
    try:
        return Decimal(raw)
    except InvalidOperation:
        raise SystemExit(f"Invalid price: {raw!r}") from None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Live evaluation against the real model and embeddings (paid).")
    parser.add_argument("--confirm-live", action="store_true",
                        help="Required. Acknowledges that this spends API credit.")
    parser.add_argument("--case", help="Comma-separated case ids to run.")
    parser.add_argument("--save", action="store_true",
                        help=f"Write {LIVE_REPORT.relative_to(REPORTS_DIR.parent.parent)}")
    parser.add_argument("--compare", action="store_true",
                        help="Print a side-by-side comparison with the offline report.")
    parser.add_argument("--input-price", help="USD per million input tokens, overriding config.")
    parser.add_argument("--output-price", help="USD per million output tokens, overriding config.")
    args = parser.parse_args(argv)

    if not args.confirm_live:
        raise SystemExit(
            "Live evaluation makes paid API calls. Re-run with --confirm-live to proceed."
        )

    settings = Settings()
    preflight(settings)
    settings = settings.model_copy(update={
        "input_cost_per_million_usd": _price(
            args.input_price, settings.input_cost_per_million_usd),
        "output_cost_per_million_usd": _price(
            args.output_price, settings.output_cost_per_million_usd),
    })

    dataset = validate_dataset()
    cases = load_cases()
    if args.case:
        wanted = {item.strip() for item in args.case.split(",") if item.strip()}
        known = {case.id for case in cases}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"Unknown case ids: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.id in wanted]

    print(f"LIVE EVALUATION: {len(cases)} case(s) against {settings.chat_model} "
          f"+ {settings.embedding_model}.")
    print("Each case makes one decision call, up to one composition call, and one "
          "embedding query. This spends real API credit.")

    with tempfile.TemporaryDirectory() as folder:
        database = Database(Path(folder) / "live.sqlite3")
        database.initialize()
        seed_database(database, dataset)
        pipeline, sink = build_live_pipeline(settings, database)

        def run_one(case: LiveCase) -> tuple[PipelineResponse, TraceRecord]:
            response = pipeline.run(
                PipelineRequest(message=case.message),
                SessionContext(customer_id=case.customer_id),
            )
            return response, sink.records[-1]

        rows = run_cases(cases, dataset, run_one)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "retrieval_top_k": settings.retrieval_top_k,
        **summarize(rows),
        "results": rows,
    }

    if args.save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        LIVE_REPORT.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nSaved {LIVE_REPORT}")

    if args.compare:
        if OFFLINE_REPORT.is_file():
            _print_comparison(compare_reports(
                json.loads(OFFLINE_REPORT.read_text()), report))
        else:
            print("\nNo offline report found. Run `python -m evaluation.pipeline --save` first.")

    for row in rows:
        marker = "PASS" if row["passed"] else "FAIL"
        failed = [name for name, ok in row["checks"].items() if not ok]
        detail = f" failed={failed}" if failed else ""
        print(f"  [{marker}] {row['case_id']:<28} status={row['status']:<22}{detail}")

    print(json.dumps({key: report[key] for key in (
        "cases", "passed", "pass_rate", "status_accuracy", "source_accuracy",
        "tool_accuracy", "authorization_violations", "unsupported_claims",
        "citation_violations", "retrieval", "latency_ms", "tokens",
        "estimated_cost_usd")}, indent=2))

    if report["authorization_violations"] or report["unsupported_claims"] \
            or report["citation_violations"]:
        raise SystemExit("Live evaluation found a safety violation.")


if __name__ == "__main__":
    main()
