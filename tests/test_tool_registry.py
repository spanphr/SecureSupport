import unittest

from app.schemas.pipeline import SessionContext
from app.storage.repository import SupportRepository
from app.tools.registry import (
    READ_TOOL_NAMES,
    TOOL_SPECS,
    WRITE_TOOL_NAMES,
    ToolExecutor,
    catalog,
)
from evaluation.pipeline import FlakyRepository
from tests.helpers import harness


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.harness = harness(self)
        self.executor = ToolExecutor(self.harness.tools)
        self.alex = SessionContext(customer_id="usr_001")
        self.sam = SessionContext(customer_id="usr_002")

    def test_catalog_is_an_explicit_allowlist(self):
        names = {spec["name"] for spec in catalog()}
        self.assertEqual(names, {"get_account_status", "get_payment_status",
                                 "get_ticket_status", "create_support_ticket"})
        self.assertEqual(set(TOOL_SPECS), names)
        self.assertEqual(READ_TOOL_NAMES | WRITE_TOOL_NAMES, names)
        self.assertFalse(READ_TOOL_NAMES & WRITE_TOOL_NAMES)

    def test_unknown_and_restricted_operations_never_reach_a_handler(self):
        for name in ("issue_refund", "delete_account", "get_payment", ""):
            result = self.executor.execute(name=name, arguments={}, session=self.sam)
            self.assertEqual(result.outcome.code, "unknown_tool", name)
            self.assertIsNone(result.outcome.payload)

    def test_arguments_are_validated_before_execution(self):
        for arguments in ({}, {"record_id": ""}, {"record_id": "pay_002", "user_id": "usr_001"}):
            result = self.executor.execute(name="get_payment_status", arguments=arguments,
                                           session=self.sam)
            self.assertEqual(result.outcome.code, "invalid_arguments", arguments)
        no_args = self.executor.execute(name="get_account_status",
                                        arguments={"record_id": "pay_002"}, session=self.sam)
        self.assertEqual(no_args.outcome.code, "invalid_arguments")

    def test_owner_can_read_own_records(self):
        result = self.executor.execute(name="get_payment_status",
                                       arguments={"record_id": "pay_002"}, session=self.sam)
        self.assertEqual(result.outcome.code, "ok")
        self.assertEqual(result.outcome.payload["amount_cents"], 2500)
        self.assertEqual(result.outcome.payload["user_id"], "usr_002")
        self.assertGreater(result.duration_ms, 0)

    def test_foreign_and_missing_records_are_indistinguishable(self):
        foreign = self.executor.execute(name="get_payment_status",
                                        arguments={"record_id": "pay_002"}, session=self.alex)
        missing = self.executor.execute(name="get_payment_status",
                                        arguments={"record_id": "pay_404"}, session=self.alex)
        self.assertEqual(foreign.outcome.code, "not_found")
        self.assertEqual(missing.outcome.code, "not_found")
        self.assertEqual(foreign.outcome.message, missing.outcome.message)
        self.assertIsNone(foreign.outcome.payload)

    def test_account_status_exposes_no_profile_data(self):
        result = self.executor.execute(name="get_account_status", arguments={}, session=self.alex)
        self.assertEqual(result.outcome.code, "ok")
        self.assertEqual(result.outcome.payload["user_id"], "usr_001")
        self.assertTrue(result.outcome.payload["mfa_enabled"])
        self.assertNotIn("email", result.outcome.payload)
        self.assertNotIn("name", result.outcome.payload)

    def test_writes_require_confirmation_and_persist_when_confirmed(self):
        arguments = {"subject": "Refund follow-up", "message": "Please check my refund."}
        blocked = self.executor.execute(name="create_support_ticket", arguments=arguments,
                                        session=self.alex)
        self.assertEqual(blocked.outcome.code, "confirmation_required")
        self.assertEqual(self.harness.repository.list_complaints("usr_001"), [])

        created = self.executor.execute(name="create_support_ticket", arguments=arguments,
                                        session=self.alex, confirmed=True)
        self.assertEqual(created.outcome.code, "ok")
        ticket_id = created.outcome.payload["id"]
        stored = SupportRepository(self.harness.database).get_ticket("usr_001", ticket_id)
        self.assertEqual(stored.status, "open")
        self.assertEqual(stored.subject, "Refund follow-up")

    def test_handler_failures_become_structured_outcomes(self):
        broken = harness(self, repository_factory=FlakyRepository)
        result = ToolExecutor(broken.tools).execute(
            name="get_payment_status", arguments={"record_id": "pay_002"},
            session=SessionContext(customer_id="usr_002"))
        self.assertEqual(result.outcome.code, "execution_failed")
        self.assertIsNone(result.outcome.payload)
        self.assertNotIn("simulated storage failure", result.outcome.message)

    def test_outcomes_encode_their_own_invariant(self):
        ok = self.executor.execute(name="get_account_status", arguments={}, session=self.alex)
        self.assertTrue(ok.outcome.ok)
        self.assertIsNotNone(ok.outcome.payload)


if __name__ == "__main__":
    unittest.main()
