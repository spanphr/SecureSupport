"""Contract tests for the OpenAI model adapter.

These assert the SDK usage and structured-output wiring with a mocked client; they
make no network calls. Orchestration behaviour lives in test_pipeline.py.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.model import OpenAISupportModel
from app.config import Settings
from app.schemas.decision import AgentDecisionDraft
from app.schemas.pipeline import ConversationTurn, GroundedAnswer
from app.schemas.retrieval import RetrievalHit
from tests.helpers import decision, tool_call


def settings(**overrides):
    return Settings(_env_file=None, openai_api_key="test-only-placeholder", **overrides)


class ModelAdapterTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("app.agent.model.OpenAI")
        self.sdk = patcher.start()
        self.addCleanup(patcher.stop)
        self.model = OpenAISupportModel(settings())

    def test_missing_key_is_rejected_before_any_call(self):
        with self.assertRaisesRegex(ValueError, "SECURESUPPORT_OPENAI_API_KEY"):
            OpenAISupportModel(Settings(_env_file=None, openai_api_key=None))
        with self.assertRaisesRegex(ValueError, "SECURESUPPORT_OPENAI_API_KEY"):
            OpenAISupportModel(Settings(_env_file=None, openai_api_key="   "))

    def test_decide_uses_structured_output_without_remote_storage(self):
        parsed = decision()
        self.sdk.return_value.responses.parse.return_value = SimpleNamespace(
            output_parsed=parsed, usage=SimpleNamespace(input_tokens=70, output_tokens=25))

        result = self.model.decide(message="Refund policy?", history=[], tools=[{"name": "x"}])

        self.assertEqual(result.value, parsed)
        self.assertEqual(result.usage.input_tokens, 70)
        self.assertEqual(result.usage.output_tokens, 25)
        arguments = self.sdk.return_value.responses.parse.call_args.kwargs
        self.assertFalse(arguments["store"])
        self.assertIs(arguments["text_format"], AgentDecisionDraft)
        self.assertEqual(arguments["model"], "gpt-4.1-mini")
        payload = json.loads(arguments["input"][1]["content"])
        self.assertEqual(payload["message"], "Refund policy?")
        self.assertEqual(payload["tools"], [{"name": "x"}])

    def test_stray_tool_fields_from_the_model_are_discarded(self):
        # Structured outputs require every field, so models fill tool fields even
        # for tool-free actions. That must not fail or widen the decision.
        draft = AgentDecisionDraft(
            category="refunds", action="answer_policy", needs_retrieval=True,
            needs_tool=False, needs_clarification=False, needs_escalation=False,
            tool_name="get_payment_status", tool_arguments={"record_id": "pay_001"},
        )
        self.sdk.return_value.responses.parse.return_value = SimpleNamespace(
            output_parsed=draft, usage=None)
        result = self.model.decide(message="Refund policy?", history=[], tools=[])
        self.assertEqual(result.value.action, "answer_policy")
        self.assertIsNone(result.value.tool_name)
        self.assertIsNone(result.value.tool_arguments)
        self.assertFalse(result.value.needs_tool)

    def test_tool_name_is_derived_from_the_action_not_from_the_model(self):
        draft = AgentDecisionDraft(
            category="payments", action="lookup_payment", needs_retrieval=False,
            needs_tool=True, needs_clarification=False, needs_escalation=False,
            tool_name="delete_account", tool_arguments={"record_id": "pay_002"},
        )
        self.sdk.return_value.responses.parse.return_value = SimpleNamespace(
            output_parsed=draft, usage=None)
        result = self.model.decide(message="Status of pay_002?", history=[], tools=[])
        self.assertEqual(result.value.tool_name, "get_payment_status")

    def test_compose_receives_articles_and_confirmed_tool_results(self):
        hit = RetrievalHit(article_id="kb_001", title="Refund requests",
                           content="Requires approval.", distance=0.2)
        call = tool_call("get_payment_status", "ok",
                         {"id": "pay_002", "user_id": "usr_002", "status": "pending"})
        self.sdk.return_value.responses.parse.return_value = SimpleNamespace(
            output_parsed=GroundedAnswer(answer="Approval required.", source_ids=["kb_001"],
                                         supported=True),
            usage=None)

        result = self.model.compose(
            message="Refund?", history=[ConversationTurn(role="customer", content="hi")],
            hits=[hit], tool_calls=[call])

        self.assertTrue(result.value.supported)
        self.assertEqual(result.usage.input_tokens, 0)
        arguments = self.sdk.return_value.responses.parse.call_args.kwargs
        self.assertIs(arguments["text_format"], GroundedAnswer)
        payload = json.loads(arguments["input"][1]["content"])
        self.assertEqual(payload["articles"][0]["id"], "kb_001")
        self.assertEqual(payload["tool_results"][0]["result"]["status"], "pending")
        self.assertEqual(payload["recent_turns"][0]["role"], "customer")

    def test_missing_structured_output_is_a_failure_not_a_silent_answer(self):
        self.sdk.return_value.responses.parse.return_value = SimpleNamespace(
            output_parsed=None, usage=None)
        with self.assertRaisesRegex(ValueError, "No structured decision"):
            self.model.decide(message="Hi", history=[], tools=[])
        with self.assertRaisesRegex(ValueError, "No structured grounded answer"):
            self.model.compose(message="Hi", history=[], hits=[], tool_calls=[])

    def test_sdk_errors_propagate_for_the_pipeline_to_contain(self):
        self.sdk.return_value.responses.parse.side_effect = RuntimeError("upstream 500")
        with self.assertRaises(RuntimeError):
            self.model.decide(message="Hi", history=[], tools=[])


if __name__ == "__main__":
    unittest.main()
