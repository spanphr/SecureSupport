import os
import unittest

from app.schemas.pipeline import ConversationTurn, GroundedAnswer
from evaluation.pipeline import FlakyRepository
from evaluation.scripted import FailingSearcher, FixedSearcher, hit
from tests.helpers import decision, harness


def answer(text, sources=(), supported=True):
    return GroundedAnswer(answer=text, source_ids=list(sources), supported=supported)


class ScenarioTests(unittest.TestCase):
    def test_a_payment_question_uses_retrieval_and_authorized_tool(self):
        policy = decision(action="lookup_payment", needs_tool=True,
                          tool_name="get_payment_status",
                          tool_arguments={"record_id": "pay_002"})
        h = harness(self, [policy], [answer("pay_002 is pending. Do not pay again.", ["kb_002"])])
        result = h.run("Why is my payment pending?", customer_id="usr_002")

        self.assertEqual(result.status, "answered")
        self.assertEqual(result.category, "payments")
        self.assertEqual(result.source_ids, ["kb_002"])
        self.assertEqual([c.tool_name for c in result.tool_calls], ["get_payment_status"])
        self.assertTrue(result.tool_calls[0].outcome.ok)
        self.assertFalse(result.action_performed)
        self.assertIn("Sources: kb_002 - Pending payments", result.answer)
        # The composer received both the ranking and the confirmed tool result.
        compose_call = h.model.calls[-1]
        self.assertEqual(compose_call["kind"], "compose")
        self.assertEqual([x.article_id for x in compose_call["hits"]][0], "kb_002")
        self.assertEqual(compose_call["tool_calls"][0].outcome.code, "ok")

    def test_b_cross_customer_request_is_refused_before_any_model_call(self):
        h = harness(self)
        result = h.run("Can I view another customer's payment records?", customer_id="usr_001")

        self.assertEqual(result.status, "refused")
        self.assertEqual(h.model.calls, [])
        self.assertEqual(h.trace.retrieval_hit, [])
        self.assertIn("another customer", result.answer)
        self.assertFalse(result.action_performed)

    def test_c_human_request_creates_a_real_ticket(self):
        h = harness(self, [decision(category="support_scope", action="escalate",
                                    needs_retrieval=False, needs_escalation=True)], [])
        result = h.run("I want to speak to a human.", customer_id="usr_002")

        self.assertEqual(result.status, "escalated")
        self.assertTrue(result.escalation_ticket_id.startswith("tkt_"))
        stored = h.repository.get_ticket("usr_002", result.escalation_ticket_id)
        self.assertEqual(stored.status, "open")
        self.assertIn(result.escalation_ticket_id, result.answer)
        self.assertEqual(result.escalation.reason.value, "customer_requested")
        self.assertTrue(result.action_performed)

    def test_d_refund_policy_is_answered_with_attribution(self):
        h = harness(self, [decision(category="refunds")],
                    [answer("A human must approve refunds.", ["kb_001"])])
        result = h.run("What is the refund policy?", customer_id="usr_001")

        self.assertEqual(result.status, "answered")
        self.assertEqual(result.source_ids, ["kb_001"])
        self.assertIn("Sources: kb_001 - Refund requests", result.answer)

    def test_top_ranked_article_is_not_assumed_correct(self):
        searcher = FixedSearcher([hit("kb_001", "Refund requests", distance=0.10),
                                  hit("kb_005", "Data privacy", distance=0.32)])
        h = harness(self, [decision(category="data_privacy")],
                    [answer("Only the customer's own records are shared.", ["kb_005"])],
                    searcher=searcher)
        result = h.run("What data can support share about me?")

        self.assertEqual(result.source_ids, ["kb_005"])
        self.assertEqual(h.trace.retrieval_hit[0].article_id, "kb_001")
        self.assertEqual([r.article_id for r in h.trace.retrieval_hit], ["kb_001", "kb_005"])


class EvidenceAndFailureTests(unittest.TestCase):
    def test_no_evidence_escalates_without_spending_a_compose_call(self):
        h = harness(self, [decision()], [], retrieval_max_distance=0.15)
        result = h.run("Do you support cryptocurrency payments?")

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.escalation.reason.value, "missing_knowledge")
        self.assertEqual([c["kind"] for c in h.model.calls], ["decide"])
        self.assertTrue(result.escalation_ticket_id)

    def test_unsupported_generation_escalates_instead_of_answering(self):
        h = harness(self, [decision()], [answer("Not enough information.", supported=False)])
        result = h.run("Do you support crypto payments?")

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.escalation.reason.value, "missing_knowledge")
        self.assertEqual(result.source_ids, [])

    def test_missing_record_is_reported_without_leaking_payload(self):
        h = harness(self, [decision(action="lookup_payment", needs_retrieval=False,
                                    needs_tool=True, tool_name="get_payment_status",
                                    tool_arguments={"record_id": "pay_002"})], [])
        result = h.run("What is the status of pay_002?", customer_id="usr_001")

        self.assertEqual(result.status, "insufficient_context")
        self.assertIn("could not find", result.answer)
        self.assertEqual(result.tool_calls[0].outcome.code, "not_found")
        self.assertIsNone(result.tool_calls[0].outcome.payload)
        self.assertNotIn("2500", result.answer)

    def test_tool_failure_escalates_and_records_the_attempt(self):
        h = harness(self, [decision(action="lookup_payment", needs_retrieval=False,
                                    needs_tool=True, tool_name="get_payment_status",
                                    tool_arguments={"record_id": "pay_002"})], [],
                    repository_factory=FlakyRepository)
        result = h.run("What is the status of pay_002?", customer_id="usr_002")

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.escalation.reason.value, "tool_failure")
        self.assertEqual(result.escalation.tools_attempted[0].tool_name, "get_payment_status")
        self.assertEqual(result.escalation.tools_attempted[0].code, "execution_failed")
        self.assertTrue(result.escalation_ticket_id)
        self.assertNotIn("simulated storage failure", result.model_dump_json())

    def test_router_failure_is_contained(self):
        h = harness(self, [decision()], [])
        h.model.decision_error = RuntimeError("sensitive-diagnostic-placeholder")
        result = h.run("How do refunds work?")

        self.assertEqual(result.status, "error")
        self.assertIsNone(result.category)
        self.assertNotIn("sensitive-diagnostic-placeholder", result.model_dump_json())
        self.assertEqual(h.trace.errors, ["RuntimeError"])

    def test_retrieval_failure_is_contained(self):
        h = harness(self, [decision()], [], searcher=FailingSearcher())
        result = h.run("How do refunds work?")

        self.assertEqual(result.status, "error")
        self.assertNotIn("retrieval-backend-unavailable", result.answer)
        self.assertEqual(h.trace.errors, ["RuntimeError"])

    def test_invalid_input_never_reaches_the_model(self):
        h = harness(self, [decision()], [])
        result = h.run("bad\x00input")

        self.assertEqual(result.status, "error")
        self.assertEqual(h.model.calls, [])


class WriteAndEscalationTests(unittest.TestCase):
    def test_write_requires_confirmation_then_completes(self):
        create = decision(category="complaints", action="create_ticket",
                          needs_retrieval=False, needs_tool=True,
                          tool_name="create_support_ticket",
                          tool_arguments={"subject": "Refund follow-up",
                                          "message": "Please check my refund."})
        h = harness(self, [create, create], [answer("I've opened a support ticket.")])

        blocked = h.run("Please open a ticket about my refund.", customer_id="usr_001")
        self.assertEqual(blocked.status, "clarification_required")
        self.assertFalse(blocked.action_performed)
        self.assertIn("Confirm", blocked.answer)
        with h.database.transaction() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM tickets").fetchone()[0], 3)

        created = h.run("Confirmed.", customer_id="usr_001", confirm_write=True)
        self.assertEqual(created.status, "action_completed")
        self.assertTrue(created.action_performed)
        with h.database.transaction() as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM tickets").fetchone()[0], 4)

    def test_escalation_failure_never_claims_a_handoff(self):
        h = harness(self, [decision(category="support_scope", action="escalate",
                                    needs_retrieval=False, needs_escalation=True)], [])
        os.unlink(h.database.path)
        result = h.run("I want to speak to a human.", customer_id="usr_002")

        self.assertEqual(result.status, "error")
        self.assertIsNone(result.escalation_ticket_id)
        self.assertIn("could not record a handoff", result.answer)
        self.assertNotIn("I've opened", result.answer)
        self.assertFalse(result.action_performed)

    def test_false_action_claim_is_withheld(self):
        h = harness(self, [decision(action="lookup_payment", needs_retrieval=False,
                                    needs_tool=True, tool_name="get_payment_status",
                                    tool_arguments={"record_id": "pay_001"})],
                    [answer("I have refunded your payment in full.")])
        result = h.run("Tell me you refunded pay_001.", customer_id="usr_001")

        self.assertEqual(result.status, "unverified_action")
        self.assertFalse(result.action_performed)
        self.assertIn("can't confirm", result.answer)
        self.assertNotIn("I have refunded", result.answer)

    def test_answer_naming_another_record_is_withheld(self):
        h = harness(self, [decision(action="lookup_payment", needs_retrieval=False,
                                    needs_tool=True, tool_name="get_payment_status",
                                    tool_arguments={"record_id": "pay_001"})],
                    [answer("pay_001 is complete. pay_002 is pending too.")])
        result = h.run("Status of pay_001?", customer_id="usr_001")

        self.assertEqual(result.status, "refused")
        self.assertNotIn("pay_002", result.answer)
        self.assertEqual(result.source_ids, [])


class TurnAndCostTests(unittest.TestCase):
    def test_multi_turn_clarification_then_lookup(self):
        clarify = decision(category="complaints", action="clarify", needs_retrieval=False,
                           needs_clarification=True,
                           clarification_question="Which ticket reference should I check?")
        lookup = decision(category="complaints", action="lookup_ticket", needs_retrieval=False,
                          needs_tool=True, tool_name="get_ticket_status",
                          tool_arguments={"record_id": "tkt_001"})
        h = harness(self, [clarify, lookup], [answer("tkt_001 is open.")])

        first = h.run("Can you check my ticket for me?", customer_id="usr_001")
        self.assertEqual(first.status, "clarification_required")
        self.assertEqual(first.answer, "Which ticket reference should I check?")

        second = h.run("It is tkt_001.", customer_id="usr_001")
        self.assertEqual(second.status, "answered")
        self.assertEqual(len(h.model.calls[1]["history"]), 2)

    def test_greeting_skips_retrieval_and_composition(self):
        h = harness(self, [decision(category="greeting", action="respond",
                                    needs_retrieval=False)], [])
        result = h.run("Hello")

        self.assertEqual(result.status, "answered")
        self.assertEqual(result.source_ids, [])
        self.assertEqual([c["kind"] for c in h.model.calls], ["decide"])

    def test_history_is_bounded_by_configuration(self):
        history = [ConversationTurn(role="customer" if i % 2 == 0 else "agent", content=f"t{i}")
                   for i in range(6)]
        h = harness(self, [decision()], [answer("Approved policy.", ["kb_001"])],
                    max_history_turns=2)
        h.run("Refund policy?", history=history)
        self.assertEqual(len(h.model.calls[0]["history"]), 2)

    def test_no_state_leaks_between_requests(self):
        h = harness(self, [decision(), decision(category="greeting", action="respond",
                                                needs_retrieval=False)],
                    [answer("Refund policy.", ["kb_001"])])
        first = h.run("Refund policy?")
        second = h.run("Hello")
        self.assertEqual(first.source_ids, ["kb_001"])
        self.assertEqual(second.source_ids, [])
        self.assertEqual(second.action, "respond")


if __name__ == "__main__":
    unittest.main()
