"""Offline tests for the live evaluation harness.

The harness itself makes paid API calls, so these tests only exercise its pure
scoring, aggregation, and spend guards with fabricated responses.
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from app.config import Settings
from app.observability import TraceRecord
from app.schemas.pipeline import EvidenceRef, PipelineResponse
from evaluation import live
from evaluation.live import (
    LiveCase,
    LiveTurn,
    compare_reports,
    load_cases,
    run_cases,
    score_case,
    score_conversation,
    summarize,
)
from scripts.validate_data import validate_dataset
from tests.helpers import tool_call


def trace(ids=(), input_tokens=10, output_tokens=5, cost=0.001):
    return TraceRecord(
        trace_id="trace", started_at=datetime.now(timezone.utc),
        message_fingerprint="fingerprint", message_chars=12,
        retrieval_hit=[
            EvidenceRef(article_id=article, title=article, rank=rank, distance=0.1)
            for rank, article in enumerate(ids, 1)
        ],
        input_tokens=input_tokens, output_tokens=output_tokens,
        estimated_cost_usd=cost,
    )


def response(status="answered", answer="Grounded answer.", sources=(), tools=(),
             ticket=None, performed=False):
    return PipelineResponse(
        trace_id="trace", status=status, answer=answer, source_ids=list(sources),
        tool_calls=list(tools), action_performed=performed,
        escalation_ticket_id=ticket, total_latency_ms=12.5,
    )


CASE = LiveCase(
    id="c1", scenario="scenario", customer_id="usr_001", message="Refund policy?",
    allowed_statuses=["answered"], required_source=["kb_001"],
    must_not_contain=["2500"],
)
TOOL_CASE = CASE.model_copy(
    update={"id": "c2", "required_source": [], "expected_tool": "get_payment_status"})


class LiveCaseTests(unittest.TestCase):
    def test_case_file_is_valid_and_references_real_customers(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 8)
        self.assertEqual(len({case.id for case in cases}), len(cases))
        users = {user.id for user in validate_dataset().users}
        for case in cases:
            self.assertTrue(case.scripted_turns, case.id)
            for turn in case.scripted_turns:
                self.assertTrue(turn.message.strip(), case.id)
                self.assertTrue(turn.allowed_statuses, case.id)
            self.assertIn(case.customer_id, users, case.id)

    def test_allowed_statuses_cannot_be_empty(self):
        with self.assertRaises(Exception):
            LiveCase(id="x", scenario="s", customer_id="usr_001", message="m",
                     allowed_statuses=[])


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.dataset = validate_dataset()

    def check(self, case=CASE, resp=None, tr=None):
        return score_case(case, resp or response(answer="Approved policy.", sources=["kb_001"]),
                          tr or trace(["kb_001", "kb_002"]), self.dataset)

    def test_a_clean_response_passes(self):
        row = self.check()
        self.assertTrue(row["passed"], row["checks"])
        self.assertEqual(row["sources"], ["kb_001"])

    def test_wrong_status_fails(self):
        row = self.check(resp=response(status="refused", answer="No.", sources=["kb_001"]))
        self.assertFalse(row["checks"]["status"])
        self.assertFalse(row["passed"])

    def test_missing_required_source_fails(self):
        row = self.check(resp=response(answer="Approved.", sources=["kb_002"]))
        self.assertFalse(row["checks"]["sources"])

    def test_unrequested_tool_fails_when_one_was_expected(self):
        row = self.check(case=TOOL_CASE,
                         resp=response(answer="Approved.", sources=["kb_001"]))
        self.assertFalse(row["checks"]["tool"])

    def test_expected_tool_satisfied_passes(self):
        row = self.check(
            case=TOOL_CASE,
            resp=response(answer="pay_001 is complete.", sources=[],
                          tools=[tool_call("get_payment_status", "ok",
                                           {"id": "pay_001", "user_id": "usr_001"})]),
            tr=trace())
        self.assertTrue(row["checks"]["tool"])
        self.assertTrue(row["passed"], row["checks"])

    def test_foreign_customer_text_is_a_violation(self):
        row = self.check(resp=response(answer="pay_002 is pending.", sources=["kb_001"]))
        self.assertFalse(row["checks"]["no_authorization_leak"])
        self.assertFalse(row["passed"])

    def test_unconfirmed_completion_claim_is_a_violation(self):
        row = self.check(resp=response(answer="I have refunded your payment.", sources=["kb_001"]))
        self.assertFalse(row["checks"]["no_unsupported_claim"])

    def test_confirmed_write_makes_a_claim_legitimate(self):
        created = tool_call("create_support_ticket", "ok",
                            {"id": "tkt_new", "user_id": "usr_001", "status": "open"})
        case = CASE.model_copy(update={"required_source": [], "expected_tool": None,
                                       "allowed_statuses": ["action_completed"]})
        row = self.check(case=case,
                         resp=response(status="action_completed", performed=True,
                                       answer="I've opened support ticket tkt_new.",
                                       tools=[created]))
        self.assertTrue(row["passed"], row["checks"])

    def test_forbidden_text_is_detected(self):
        row = self.check(resp=response(answer="The balance is 2500 cents.", sources=["kb_001"]))
        self.assertFalse(row["checks"]["no_forbidden_text"])

    def test_invented_citation_is_a_violation(self):
        row = self.check(resp=response(answer="Approved.", sources=["kb_999"]))
        self.assertFalse(row["checks"]["citations"])


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = validate_dataset()
        self.cases = [
            LiveCase(id="a", scenario="s", customer_id="usr_002", message="m1",
                     allowed_statuses=["answered"], required_source=["kb_002"]),
            LiveCase(id="b", scenario="s", customer_id="usr_002", message="m2",
                     allowed_statuses=["answered"], expected_tool="get_payment_status"),
            LiveCase(id="c", scenario="s", customer_id="usr_001", message="m3",
                     allowed_statuses=["refused"]),
        ]

    def test_run_cases_and_summary(self):
        scripted = {
            "a": (response(answer="Pending.", sources=["kb_002"]), trace(["kb_002"])),
            "b": (response(answer="pay_002 is pending.",
                           tools=[tool_call("get_payment_status", "ok",
                                            {"id": "pay_002", "user_id": "usr_002"})]),
                  trace()),
            "c": (response(status="refused", answer="I can't share that."), trace()),
        }
        rows = run_cases(self.cases, self.dataset, lambda case: [scripted[case.id]])
        summary = summarize(rows)

        self.assertEqual(summary["cases"], 3)
        self.assertEqual(summary["passed"], 3)
        self.assertEqual(summary["pass_rate"], 1.0)
        self.assertEqual(summary["tool_accuracy"], 1.0)
        self.assertEqual(summary["retrieval"]["hit_at_1"], 1.0)
        self.assertEqual(summary["retrieval"]["hit_at_3"], 1.0)
        self.assertEqual(summary["authorization_violations"], 0)
        self.assertAlmostEqual(summary["latency_ms"]["mean"], 12.5)
        self.assertAlmostEqual(summary["estimated_cost_usd"], 0.003)

    def test_cost_is_unknown_when_any_case_is_unpriced(self):
        rows = run_cases(self.cases, self.dataset,
                         lambda case: [(response(answer="ok", sources=["kb_002"])
                                        if case.id == "a"
                                        else response(status="refused", answer="no"),
                                        trace(["kb_002"] if case.id == "a" else (),
                                              cost=None if case.id == "a" else 0.001))])
        self.assertIsNone(summarize(rows)["estimated_cost_usd"])

    def test_rates_exclude_cases_without_the_dimension(self):
        scripted = {
            "a": (response(answer="Pending.", sources=["kb_002"]), trace(["kb_002"])),
            "b": (response(answer="No tool ran."), trace()),
            "c": (response(status="refused", answer="no"), trace()),
        }
        rows = run_cases(self.cases, self.dataset, lambda case: [scripted[case.id]])
        summary = summarize(rows)
        self.assertEqual(summary["tool_accuracy"], 0.0)
        self.assertEqual(summary["retrieval"]["cases"], 1)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])


class MultiTurnTests(unittest.TestCase):
    """A conversation passes only when every turn passes."""

    def setUp(self):
        self.dataset = validate_dataset()
        self.case = LiveCase(
            id="conv", scenario="conversation", customer_id="usr_002",
            turns=[
                LiveTurn(message="Why is my payment pending?",
                         allowed_statuses=["clarification_required", "answered"]),
                LiveTurn(message="It is pay_002.", allowed_statuses=["answered"],
                         expected_tool="get_payment_status"),
            ],
        )
        self.lookup = response(
            answer="pay_002 is pending.",
            tools=[tool_call("get_payment_status", "ok",
                             {"id": "pay_002", "user_id": "usr_002"})])
        self.clarify = response(status="clarification_required", answer="Which payment?")

    def test_a_case_is_either_one_message_or_a_conversation(self):
        with self.assertRaises(Exception):
            LiveCase(id="x", scenario="s", customer_id="usr_001", message="m",
                     allowed_statuses=["answered"], turns=self.case.turns)
        with self.assertRaises(Exception):
            LiveCase(id="x", scenario="s", customer_id="usr_001")

    def test_clean_conversation_passes(self):
        row = score_conversation(
            self.case, [(self.clarify, trace()), (self.lookup, trace())], self.dataset)
        self.assertTrue(row["passed"], row["checks"])
        self.assertEqual(row["turn_count"], 2)
        self.assertTrue(row["expects_tool"])

    def test_bad_first_turn_fails_the_case(self):
        row = score_conversation(
            self.case,
            [(response(status="refused", answer="No."), trace()), (self.lookup, trace())],
            self.dataset)
        self.assertFalse(row["checks"]["status"])
        self.assertFalse(row["passed"])

    def test_safety_violation_in_any_turn_fails_the_case(self):
        leak = response(status="clarification_required",
                        answer="pay_001 belongs to usr_001.")
        row = score_conversation(
            self.case, [(leak, trace()), (self.lookup, trace())], self.dataset)
        self.assertFalse(row["checks"]["no_authorization_leak"])
        self.assertFalse(row["passed"])

    def test_missing_turn_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 2 turn result"):
            score_conversation(self.case, [(self.lookup, trace())], self.dataset)

    def test_totals_are_summed_over_turns(self):
        row = score_conversation(
            self.case,
            [(self.clarify, trace(cost=0.002)), (self.lookup, trace(cost=0.003))],
            self.dataset)
        self.assertAlmostEqual(row["estimated_cost_usd"], 0.005)
        self.assertAlmostEqual(row["latency_ms"], 25.0)
        self.assertEqual(row["input_tokens"], 20)

    def test_single_turn_shorthand_still_scores(self):
        row = score_conversation(
            CASE, [(response(answer="Approved policy.", sources=["kb_001"]),
                    trace(["kb_001"]))], self.dataset)
        self.assertTrue(row["passed"], row["checks"])
        self.assertEqual(row["turn_count"], 1)


class ComparisonTests(unittest.TestCase):
    def test_offline_and_live_metrics_line_up(self):
        offline = {
            "pipeline": {"cases": 15, "cases_passed": 15, "authorization_violations": 0,
                         "unsupported_claims": 0, "citation_violations": 0,
                         "mean_latency_ms": 5.6},
            "retrieval": {"hit_at_1": 0.75, "hit_at_3": 1.0},
        }
        live_report = {"cases": 9, "pass_rate": 0.888, "authorization_violations": 0,
                       "unsupported_claims": 0, "citation_violations": 0,
                       "retrieval": {"hit_at_1": 0.5, "hit_at_3": 1.0},
                       "latency_ms": {"mean": 1200.0}}
        rows = {row["metric"]: row for row in compare_reports(offline, live_report)}
        self.assertEqual(rows["pipeline_cases"]["offline"], 15)
        self.assertEqual(rows["pipeline_cases"]["live"], 9)
        self.assertAlmostEqual(rows["pass_rate"]["offline"], 1.0)
        self.assertEqual(rows["retrieval_hit_at_1"]["live"], 0.5)
        self.assertEqual(rows["mean_latency_ms"]["live"], 1200.0)


class SpendGuardTests(unittest.TestCase):
    def test_refuses_without_the_confirm_flag_and_never_builds_a_pipeline(self):
        with patch("evaluation.live.build_live_pipeline") as builder:
            with self.assertRaises(SystemExit) as raised:
                live.main([])
            builder.assert_not_called()
        self.assertIn("--confirm-live", str(raised.exception))

    def test_refuses_production_configuration(self):
        production = Settings(_env_file=None, environment="production",
                              openai_api_key="test-only-placeholder")
        with patch("evaluation.live.Settings", return_value=production):
            with patch("evaluation.live.build_live_pipeline") as builder:
                with self.assertRaises(SystemExit):
                    live.main(["--confirm-live"])
                builder.assert_not_called()

    def test_refuses_without_an_api_key(self):
        with self.assertRaises(SystemExit) as raised:
            live.preflight(Settings(_env_file=None))
        self.assertIn("OPENAI_API_KEY", str(raised.exception))

    def test_preflight_passes_with_a_key(self):
        live.preflight(Settings(_env_file=None, openai_api_key="test-only-placeholder"))


if __name__ == "__main__":
    unittest.main()
