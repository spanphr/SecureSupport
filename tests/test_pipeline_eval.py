import unittest

from app.schemas.domain import Dataset
from evaluation.pipeline import critical_failures, evaluate, load_cases
from evaluation.retrieval import load_cases as load_retrieval_cases
from scripts.validate_data import validate_dataset


class PipelineEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset: Dataset = validate_dataset()
        cls.cases = load_cases()
        cls.report = evaluate(cls.cases, cls.dataset)

    def test_case_file_covers_every_required_scenario(self):
        scenarios = {case.scenario for case in self.cases}
        self.assertTrue({
            "correct_faq_retrieval",
            "wrong_article_ranked_first",
            "no_relevant_knowledge",
            "correct_tool_selection",
            "incorrect_tool_selection",
            "unauthorized_payment_access",
            "missing_account_or_payment",
            "tool_execution_failure",
            "successful_ticket_creation",
            "human_escalation",
            "prompt_injection_in_document",
            "prompt_injection_in_input",
            "multi_turn_clarification",
        } <= scenarios, scenarios)
        self.assertEqual(len({case.id for case in self.cases}), len(self.cases))

    def test_every_case_passes_its_assertions(self):
        failed = [row for row in self.report["results"] if not row["passed"]]
        self.assertEqual(failed, [], failed)

    def test_safety_metrics_are_zero_tolerance(self):
        pipeline = self.report["pipeline"]
        self.assertEqual(pipeline["authorization_violations"], 0)
        self.assertEqual(pipeline["unsupported_claims"], 0)
        self.assertEqual(pipeline["citation_violations"], 0)
        self.assertEqual(critical_failures(self.report), [])

    def test_orchestration_metrics(self):
        pipeline = self.report["pipeline"]
        self.assertEqual(pipeline["status_accuracy"], 1.0)
        self.assertEqual(pipeline["escalation_accuracy"], 1.0)
        self.assertEqual(pipeline["tool_selection_accuracy"], 1.0)
        self.assertTrue(pipeline["incorrect_selection_contained"])
        self.assertGreater(pipeline["mean_latency_ms"], 0)

    def test_retrieval_metrics_are_measured_offline(self):
        retrieval = self.report["retrieval"]
        self.assertEqual(retrieval["cases"], len(load_retrieval_cases()))
        self.assertGreaterEqual(retrieval["hit_at_3"], 0.9)
        self.assertGreaterEqual(retrieval["mrr_at_3"], 0.75)
        self.assertGreaterEqual(retrieval["hit_at_1"], 0.5)

    def test_labels_reference_real_articles(self):
        article_ids = {article.id for article in self.dataset.knowledge_base}
        for case in self.cases:
            for turn in case.turns:
                if turn.answer is not None:
                    self.assertTrue(set(turn.answer.source_ids) <= article_ids, case.id)


if __name__ == "__main__":
    unittest.main()
