import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import ValidationError
from app.agent.graph import SupportAgent
from app.agent.model import OpenAISupportModel
from app.config import Settings
from app.schemas.retrieval import RetrievalHit, RetrievalResult
from app.schemas.support import GroundedAnswer, RoutingDecision


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.classify.return_value = RoutingDecision(route='retrieval')
        self.hit = RetrievalHit(article_id='kb_001', title='Refund requests',
                                content='Refunds require approval.', distance=0.1)
        self.searcher = Mock()
        self.searcher.search.return_value = RetrievalResult(query='refund', hits=[self.hit])
        self.model.answer.return_value = GroundedAnswer(answer='Refund requests require approval.',
                                                       source_ids=['kb_001'], supported=True)
        self.agent = SupportAgent(self.model, self.searcher)

    def test_direct_does_not_retrieve_or_generate(self):
        self.model.classify.return_value = RoutingDecision(route='direct')
        result = self.agent.run('Hello')
        self.assertEqual(result.status, 'answered')
        self.assertEqual(result.source_ids, [])
        self.assertFalse(result.action_performed)
        self.searcher.search.assert_not_called()
        self.model.answer.assert_not_called()

    def test_policy_path_returns_validated_sources(self):
        result = self.agent.run('How do refunds work?')
        self.assertEqual(result.route, 'retrieval')
        self.assertEqual(result.status, 'answered')
        self.assertEqual(result.source_ids, ['kb_001'])
        self.searcher.search.assert_called_once_with('How do refunds work?', top_k=3)
        self.model.answer.assert_called_once_with('How do refunds work?', [self.hit])

    def test_tool_route_does_not_execute_or_invent_actions(self):
        self.model.classify.return_value = RoutingDecision(route='tool')
        result = self.agent.run('Refund pay_001')
        self.assertEqual(result.status, 'action_unavailable')
        self.assertFalse(result.action_performed)
        self.assertIn('No record was accessed or changed', result.answer)
        self.searcher.search.assert_not_called()
        self.model.answer.assert_not_called()

    def test_out_of_scope_bypasses_retrieval(self):
        self.model.classify.return_value = RoutingDecision(route='out_of_scope')
        result = self.agent.run('Recommend a guaranteed investment')
        self.assertEqual(result.status, 'out_of_scope')
        self.searcher.search.assert_not_called()

    def test_no_cross_request_state(self):
        self.agent.run('How do refunds work?')
        self.model.classify.return_value = RoutingDecision(route='direct')
        result = self.agent.run('Hello')
        self.assertEqual(result.source_ids, [])
        self.assertEqual(result.route, 'direct')

    def test_invalid_messages_never_reach_model(self):
        for message in ('', '  ', 'x' * 4001):
            with self.assertRaises(ValidationError):
                self.agent.run(message)
        self.model.classify.assert_not_called()

    def test_router_failure_does_not_leak_exception(self):
        self.model.classify.side_effect = RuntimeError('sensitive-diagnostic-placeholder')
        result = self.agent.run('Hello')
        self.assertEqual(result.status, 'error')
        self.assertIsNone(result.route)
        self.assertNotIn('sensitive-diagnostic-placeholder', result.model_dump_json())
        self.searcher.search.assert_not_called()

    def test_retrieval_failure_does_not_generate(self):
        self.searcher.search.side_effect = RuntimeError('sensitive-diagnostic-placeholder')
        result = self.agent.run('Refund policy?')
        self.assertEqual(result.status, 'error')
        self.assertNotIn('sensitive-diagnostic-placeholder', result.answer)
        self.model.answer.assert_not_called()

    def test_empty_results_do_not_generate(self):
        self.searcher.search.return_value = RetrievalResult(query='q', hits=[])
        result = self.agent.run('Refund policy?')
        self.assertEqual(result.status, 'insufficient_context')
        self.model.answer.assert_not_called()

    def test_unsupported_and_invalid_citations_are_withheld(self):
        for supported, ids in [(False, []), (True, []), (True, ['invented']), (True, ['kb_001','invented'])]:
            self.model.answer.return_value = GroundedAnswer(
                answer='Unverified answer.', source_ids=ids, supported=supported)
            result = self.agent.run('Refund policy?')
            self.assertEqual(result.status, 'insufficient_context')
            self.assertEqual(result.source_ids, [])
            self.assertNotIn('Unverified answer', result.answer)

    def test_answer_failure_and_duplicate_citations(self):
        self.model.answer.side_effect = RuntimeError('sensitive-diagnostic-placeholder')
        self.assertEqual(self.agent.run('Refund policy?').status, 'error')
        self.model.answer.side_effect = None
        self.model.answer.return_value = GroundedAnswer(answer='Requires approval.',
                                                       supported=True, source_ids=['kb_001','kb_001'])
        self.assertEqual(self.agent.run('Refund policy?').source_ids, ['kb_001'])

    def test_structured_sdk_contract_and_missing_output(self):
        with patch('app.agent.model.OpenAI') as sdk:
            model = OpenAISupportModel(Settings(_env_file=None, openai_api_key='test-only-placeholder'))
            sdk.return_value.responses.parse.return_value = SimpleNamespace(output_parsed=RoutingDecision(route='retrieval'))
            self.assertEqual(model.classify('Refund policy?').route, 'retrieval')
            args = sdk.return_value.responses.parse.call_args.kwargs
            self.assertFalse(args['store'])
            self.assertIs(args['text_format'], RoutingDecision)
            sdk.return_value.responses.parse.return_value = SimpleNamespace(output_parsed=None)
            with self.assertRaises(ValueError):
                model.classify('Refund policy?')
            with self.assertRaises(ValueError):
                model.answer('Refund policy?', [self.hit])
        with self.assertRaises(ValueError):
            OpenAISupportModel(Settings(_env_file=None, openai_api_key=None))


if __name__ == '__main__':
    unittest.main()
