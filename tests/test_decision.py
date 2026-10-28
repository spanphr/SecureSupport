import unittest

from pydantic import ValidationError

from app.schemas.decision import (
    Action,
    AgentDecision,
    AgentDecisionDraft,
    Category,
    TOOL_FOR_ACTION,
    coerce_decision,
)
from tests.helpers import decision


class DecisionConsistencyTests(unittest.TestCase):
    def test_policy_answer_requires_and_allows_retrieval(self):
        approved = decision()
        self.assertEqual(approved.action, Action.ANSWER_POLICY)
        self.assertTrue(approved.needs_retrieval)
        self.assertFalse(approved.is_write)
        with self.assertRaisesRegex(ValidationError, "require retrieval"):
            decision(needs_retrieval=False)

    def test_tool_actions_must_name_their_mapped_tool(self):
        for action, tool in TOOL_FOR_ACTION.items():
            arguments = {} if tool == "get_account_status" else {"record_id": "pay_001"}
            if tool == "create_support_ticket":
                arguments = {"subject": "s", "message": "m"}
            built = decision(action=action, needs_retrieval=False, needs_tool=True,
                             tool_name=tool, tool_arguments=arguments)
            self.assertEqual(built.tool_name, tool)
            with self.assertRaisesRegex(ValidationError, "tool_name does not match"):
                decision(action=action, needs_retrieval=False, needs_tool=True,
                         tool_name="issue_refund", tool_arguments=arguments)

    def test_tool_actions_require_flag_and_arguments(self):
        with self.assertRaisesRegex(ValidationError, "needs_tool"):
            decision(action=Action.LOOKUP_PAYMENT, needs_retrieval=False,
                     tool_name="get_payment_status", tool_arguments={"record_id": "pay_001"})
        with self.assertRaisesRegex(ValidationError, "tool_arguments"):
            decision(action=Action.LOOKUP_PAYMENT, needs_retrieval=False, needs_tool=True,
                     tool_name="get_payment_status")

    def test_non_tool_actions_cannot_smuggle_a_tool(self):
        with self.assertRaisesRegex(ValidationError, "Only tool actions"):
            decision(action=Action.ANSWER_POLICY, tool_name="get_payment_status")
        with self.assertRaisesRegex(ValidationError, "Only tool actions"):
            decision(action=Action.ANSWER_POLICY, tool_arguments={"record_id": "pay_001"})

    def test_clarification_requires_a_question_and_only_then(self):
        clarified = decision(category="complaints", action=Action.CLARIFY, needs_retrieval=False,
                             needs_clarification=True, clarification_question="Which ticket?")
        self.assertEqual(clarified.clarification_question, "Which ticket?")
        with self.assertRaisesRegex(ValidationError, "require a question"):
            decision(action=Action.CLARIFY, needs_retrieval=False, needs_clarification=True)
        with self.assertRaisesRegex(ValidationError, "Only clarification"):
            decision(clarification_question="Why?")
        with self.assertRaisesRegex(ValidationError, "Only clarification"):
            decision(needs_clarification=True)

    def test_escalation_flag_must_match_action(self):
        with self.assertRaisesRegex(ValidationError, "needs_escalation"):
            decision(category="support_scope", action=Action.ESCALATE, needs_retrieval=False)
        with self.assertRaisesRegex(ValidationError, "needs_escalation"):
            decision(needs_escalation=True)

    def test_respond_is_limited_to_small_talk(self):
        greeting = decision(category=Category.GREETING, action=Action.RESPOND,
                            needs_retrieval=False)
        self.assertFalse(greeting.needs_retrieval)
        with self.assertRaisesRegex(ValidationError, "greetings and capability"):
            decision(category=Category.PAYMENTS, action=Action.RESPOND, needs_retrieval=False)
        with self.assertRaisesRegex(ValidationError, "does not need policy retrieval"):
            decision(category=Category.GREETING, action=Action.ANSWER_POLICY)

    def test_retrieval_is_rejected_for_non_evidence_actions(self):
        with self.assertRaisesRegex(ValidationError, "only valid for policy or tool"):
            decision(action=Action.RESPOND, needs_retrieval=True)
        with self.assertRaisesRegex(ValidationError, "only valid for policy or tool"):
            decision(category="support_scope", action=Action.ESCALATE,
                     needs_retrieval=True, needs_escalation=True)

    def test_a_greeting_cannot_be_a_refusal_and_extra_fields_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "greeting cannot be a refusal"):
            decision(category=Category.GREETING, action=Action.REFUSE, needs_retrieval=False)
        with self.assertRaises(ValidationError):
            decision(confidence=0.99)

    def test_write_like_actions_are_flagged(self):
        created = decision(category="complaints", action=Action.CREATE_TICKET,
                           needs_retrieval=False, needs_tool=True,
                           tool_name="create_support_ticket",
                           tool_arguments={"subject": "s", "message": "m"})
        self.assertTrue(created.is_write)
        self.assertFalse(decision(category="complaints", action=Action.LOOKUP_TICKET,
                                  needs_retrieval=False, needs_tool=True,
                                  tool_name="get_ticket_status",
                                  tool_arguments={"record_id": "tkt_001"}).is_write)


def draft(**overrides) -> AgentDecisionDraft:
    base = dict(
        category="payments", action="answer_policy", needs_retrieval=True,
        needs_tool=False, needs_clarification=False, needs_escalation=False,
    )
    base.update(overrides)
    return AgentDecisionDraft(**base)


class CoercionTests(unittest.TestCase):
    def test_tool_free_actions_discard_stray_tool_fields(self):
        built = coerce_decision(draft(tool_name="get_payment_status",
                                      tool_arguments={"record_id": "pay_001"}))
        self.assertIsNone(built.tool_name)
        self.assertIsNone(built.tool_arguments)
        self.assertFalse(built.needs_tool)

    def test_flags_and_tool_name_come_from_the_action(self):
        built = coerce_decision(draft(
            action=Action.LOOKUP_PAYMENT, needs_retrieval=True, needs_tool=False,
            tool_name="issue_refund", tool_arguments={"record_id": "pay_001"}))
        self.assertTrue(built.needs_tool)
        self.assertTrue(built.needs_retrieval)
        self.assertEqual(built.tool_name, "get_payment_status")

    def test_tool_arguments_default_to_empty_for_no_argument_tools(self):
        built = coerce_decision(draft(action=Action.LOOKUP_ACCOUNT,
                                      needs_retrieval=False, tool_arguments=None))
        self.assertEqual(built.tool_arguments, {})

    def test_contradictory_flags_are_overwritten(self):
        built = coerce_decision(draft(category="support_scope", action=Action.ESCALATE,
                                      needs_retrieval=True, needs_escalation=False))
        self.assertTrue(built.needs_escalation)
        self.assertFalse(built.needs_retrieval)

    def test_normalization_cannot_rescue_genuinely_invalid_decisions(self):
        with self.assertRaisesRegex(ValidationError, "require a question"):
            coerce_decision(draft(category="complaints", action=Action.CLARIFY,
                                  needs_retrieval=False, clarification_question=None))
        with self.assertRaisesRegex(ValidationError, "greetings and capability"):
            coerce_decision(draft(category=Category.PAYMENTS, action=Action.RESPOND,
                                  needs_retrieval=False))


if __name__ == "__main__":
    unittest.main()
