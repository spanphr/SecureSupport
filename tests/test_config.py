import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from tests.helpers import decision, harness


class SettingsTests(unittest.TestCase):
    def test_blank_optional_values_behave_as_unset(self):
        """Copying .env.example must not break startup."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "env"
            path.write_text(
                "SECURESUPPORT_OPENAI_API_KEY=\n"
                "SECURESUPPORT_RETRIEVAL_MAX_DISTANCE=\n"
                "SECURESUPPORT_INPUT_COST_PER_MILLION_USD=\n"
                "SECURESUPPORT_OUTPUT_COST_PER_MILLION_USD=\n"
                "SECURESUPPORT_RETRIEVAL_TOP_K=5\n"
            )
            settings = Settings(_env_file=path)
        self.assertIsNone(settings.retrieval_max_distance)
        self.assertIsNone(settings.input_cost_per_million_usd)
        self.assertIsNone(settings.openai_api_key)
        self.assertEqual(settings.retrieval_top_k, 5)

    def test_explicit_values_are_parsed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "env"
            path.write_text(
                "SECURESUPPORT_RETRIEVAL_MAX_DISTANCE=0.42\n"
                "SECURESUPPORT_INPUT_COST_PER_MILLION_USD=0.15\n"
            )
            settings = Settings(_env_file=path)
        self.assertEqual(settings.retrieval_max_distance, 0.42)
        self.assertEqual(settings.input_cost_per_million_usd, Decimal("0.15"))

    def test_bounds_are_enforced(self):
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, retrieval_top_k=0)
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, max_history_turns=99)
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, retrieval_max_distance=-1.0)

    def test_defaults_need_no_api_key(self):
        settings = Settings(_env_file=None)
        self.assertIsNone(settings.openai_api_key)
        self.assertEqual(settings.retrieval_top_k, 3)
        self.assertEqual(settings.max_history_turns, 6)
        self.assertTrue(settings.trace_log_enabled)

    def test_api_key_is_masked_in_representations(self):
        settings = Settings(_env_file=None, openai_api_key="sk-test-not-a-real-key")
        self.assertNotIn("sk-test-not-a-real-key", repr(settings))
        self.assertNotIn("sk-test-not-a-real-key", str(settings))


class CostBoundaryTests(unittest.TestCase):
    def test_configured_pricing_flows_into_the_trace(self):
        from app.schemas.pipeline import GroundedAnswer

        h = harness(self, [decision()], [GroundedAnswer(
            answer="Refunds need approval.", source_ids=["kb_001"], supported=True)],
            input_cost_per_million_usd=Decimal("1"), output_cost_per_million_usd=Decimal("2"))
        result = h.run("Refund policy?")
        # 120 input tokens at $1/M and 40 output tokens at $2/M.
        self.assertAlmostEqual(h.trace.estimated_cost_usd, 0.0002, places=9)
        self.assertEqual(result.status, "answered")


if __name__ == "__main__":
    unittest.main()
