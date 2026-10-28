import json
import unittest

from app.schemas.pipeline import EscalationReason, GroundedAnswer
from evaluation.scripted import FixedSearcher, hit
from tests.helpers import decision, harness


class EscalationTests(unittest.TestCase):
    def escalation_harness(self, *args, **kwargs):
        return harness(self, [decision(category="support_scope", action="escalate",
                                       needs_retrieval=False, needs_escalation=True)],
                       *args, **kwargs)

    def test_handoff_contains_everything_a_human_needs(self):
        h = self.escalation_harness()
        result = h.run("I want to speak to a human.", customer_id="usr_002")
        handoff = result.escalation

        self.assertEqual(handoff.reason, EscalationReason.CUSTOMER_REQUESTED)
        self.assertEqual(handoff.category, "support_scope")
        self.assertIn("I want to speak to a human", handoff.summary)
        self.assertTrue(handoff.detected_issue)
        # The handoff is assembled before the escalation ticket itself is written.
        self.assertEqual(handoff.tools_attempted, [])
        self.assertEqual(handoff.trace_id, result.trace_id)
        self.assertTrue(handoff.requires_human_review)
        self.assertIsNotNone(handoff.created_at)

    def test_ticket_stores_the_structured_handoff_for_the_owner(self):
        h = self.escalation_harness()
        result = h.run("I want to speak to a human.", customer_id="usr_002")
        stored = h.repository.get_ticket("usr_002", result.escalation_ticket_id)
        payload = json.loads(stored.message)

        self.assertTrue(stored.subject.startswith("Escalation: customer_requested"))
        self.assertEqual(payload["reason"], "customer_requested")
        self.assertEqual(payload["trace_id"], result.trace_id)
        self.assertEqual(payload["category"], "support_scope")

    def test_evidence_is_attached_when_retrieval_happened(self):
        searcher = FixedSearcher([hit("kb_004", "Complaint handling")])
        h = harness(self, [decision(category="complaints")],
                    [GroundedAnswer(answer="Not enough information.",
                                    source_ids=[], supported=False)],
                    searcher=searcher)
        result = h.run("How do you handle complaints?")

        self.assertEqual(result.status, "escalated")
        self.assertEqual(result.escalation.reason.value, "missing_knowledge")
        self.assertEqual([item.article_id for item in result.escalation.evidence], ["kb_004"])
        self.assertEqual(result.escalation.evidence[0].rank, 1)

    def test_customer_requested_escalation_is_persisted_once_per_request(self):
        h = self.escalation_harness()
        first = h.run("I want to speak to a human.", customer_id="usr_002")
        h2 = self.escalation_harness()
        second = h2.run("I want to speak to a human.", customer_id="usr_002")

        self.assertNotEqual(first.escalation_ticket_id, second.escalation_ticket_id)
        with h.database.transaction() as connection:
            rows = connection.execute(
                "SELECT id, status, user_id FROM tickets WHERE id=?", (first.escalation_ticket_id,)
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "open")
        self.assertEqual(rows[0]["user_id"], "usr_002")


if __name__ == "__main__":
    unittest.main()
