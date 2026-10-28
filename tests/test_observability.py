import json
import logging
import unittest
from decimal import Decimal

from app.observability import (
    JsonLogTraceSink,
    MemoryTraceSink,
    TraceBuilder,
    estimate_cost,
    fingerprint,
)
from app.schemas.pipeline import EscalationReason, GroundedAnswer, LLMUsage
from tests.helpers import decision, harness


class FingerprintAndCostTests(unittest.TestCase):
    def test_fingerprint_is_stable_short_and_non_reversible(self):
        first = fingerprint("my card ending 4242")
        self.assertEqual(first, fingerprint("my card ending 4242"))
        self.assertNotEqual(first, fingerprint("my card ending 1111"))
        self.assertEqual(len(first), 12)
        self.assertNotIn("4242", first)

    def test_cost_is_unknown_unless_both_prices_are_configured(self):
        usage = LLMUsage(input_tokens=1_000, output_tokens=500)
        self.assertIsNone(estimate_cost(usage, None, None))
        self.assertIsNone(estimate_cost(usage, Decimal("0.15"), None))
        self.assertIsNone(estimate_cost(usage, None, Decimal("0.60")))

    def test_cost_uses_explicit_pricing(self):
        usage = LLMUsage(input_tokens=1_000, output_tokens=500)
        cost = estimate_cost(usage, Decimal("0.15"), Decimal("0.60"))
        self.assertEqual(cost, Decimal("0.000450"))


class TraceTests(unittest.TestCase):
    def test_trace_records_decision_ranking_usage_and_timing(self):
        h = harness(self, [decision(category="refunds")],
                    [GroundedAnswer(answer="Refunds need approval.", source_ids=["kb_001"],
                                    supported=True)],
                    input_cost_per_million_usd=Decimal("0.15"),
                    output_cost_per_million_usd=Decimal("0.60"))
        result = h.run("What is the refund policy?")
        trace = h.trace

        self.assertEqual(trace.trace_id, result.trace_id)
        self.assertEqual(trace.status, "answered")
        self.assertEqual(trace.category, "refunds")
        self.assertEqual(trace.action, "answer_policy")
        self.assertEqual([item.article_id for item in trace.retrieval_hit][0], "kb_001")
        self.assertEqual([item.rank for item in trace.retrieval_hit], [1, 2, 3])
        self.assertEqual(trace.llm_calls, 2)
        self.assertEqual(trace.input_tokens, 120)
        self.assertEqual(trace.output_tokens, 40)
        self.assertAlmostEqual(trace.estimated_cost_usd, 0.000042, places=9)
        self.assertIsNotNone(trace.retrieval_latency_ms)
        self.assertIsNotNone(trace.llm_latency_ms)
        self.assertGreaterEqual(trace.total_latency_ms, 0)

    def test_trace_never_contains_message_text_or_answers(self):
        secret_message = "My card ending 4242 was double charged"
        h = harness(self, [decision()], [])
        h.run(secret_message)
        serialized = json.dumps(h.trace.model_dump(mode="json"))

        self.assertNotIn("4242", serialized)
        self.assertNotIn("double charged", serialized)
        self.assertNotIn(secret_message, serialized)
        self.assertIn(fingerprint(secret_message), serialized)
        self.assertEqual(h.trace.message_chars, len(secret_message))

    def test_trace_records_escalation_reason_and_error_classes_only(self):
        h = harness(self, [decision(category="support_scope", action="escalate",
                                    needs_retrieval=False, needs_escalation=True)], [])
        h.run("I want a human.", customer_id="usr_002")
        self.assertEqual(h.trace.escalation_reason, EscalationReason.CUSTOMER_REQUESTED.value)
        self.assertEqual(h.trace.errors, [])

        failing = harness(self, [decision()], [])
        failing.model.decision_error = RuntimeError("do not log this text")
        failing.run("Refund policy?")
        self.assertEqual(failing.trace.errors, ["RuntimeError"])
        self.assertNotIn("do not log this text",
                         json.dumps(failing.trace.model_dump(mode="json")))

    def test_trace_records_tool_names_and_codes_without_payloads(self):
        h = harness(self, [decision(action="lookup_payment", needs_retrieval=False,
                                    needs_tool=True, tool_name="get_payment_status",
                                    tool_arguments={"record_id": "pay_002"})],
                    [GroundedAnswer(answer="pay_002 is pending.", source_ids=[],
                                    supported=True)])
        h.run("Status of pay_002?", customer_id="usr_002")
        serialized = json.dumps(h.trace.model_dump(mode="json"))

        self.assertEqual([(t.tool_name, t.code) for t in h.trace.tool_calls],
                         [("get_payment_status", "ok")])
        self.assertNotIn("amount_cents", serialized)
        self.assertNotIn("pending", serialized)

    def test_json_sink_emits_one_parseable_line(self):
        builder = TraceBuilder("hello")
        record = builder.finish("answered")
        sink = JsonLogTraceSink()
        with self.assertLogs("securesupport.pipeline", level=logging.INFO) as captured:
            sink.emit(record)
        payload = json.loads(captured.output[0].split(":", 2)[-1].strip())
        self.assertEqual(payload["trace_id"], record.trace_id)
        self.assertEqual(MemoryTraceSink().records, [])

    def test_unpriced_models_report_cost_as_unknown(self):
        h = harness(self, [decision()], [GroundedAnswer(
            answer="Refunds need approval.", source_ids=["kb_001"], supported=True)])
        h.run("Refund policy?")
        self.assertIsNone(h.trace.estimated_cost_usd)


if __name__ == "__main__":
    unittest.main()
