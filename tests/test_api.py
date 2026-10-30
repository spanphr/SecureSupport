"""Transport tests.

The HTTP layer is allowed to add identity, history, and serialization - never policy.
These tests pin that: a forged cookie is rejected, a body-supplied customer id cannot
override the session, and the scripted demo still refuses and escalates exactly as the
pipeline does.
"""

import json
import unittest

from fastapi.testclient import TestClient

from app.api.identity import COOKIE_NAME, DemoIdentitySigner
from app.api.main import ConversationStore, create_app
from app.config import Settings
from app.schemas.pipeline import ConversationTurn

PAYMENT_QUESTION = "Why is my payment pending?"
CROSS_CUSTOMER = "Can I view another customer's payment records?"
HUMAN_REQUEST = "I want to speak to a human."


class ApiTestCase(unittest.TestCase):
    mode = "scripted"

    def setUp(self):
        self.app = create_app(mode=self.mode, settings=self.settings())
        # The context manager runs the lifespan, which closes the disposable database.
        self.client = self._enter(TestClient(self.app))

    def settings(self) -> Settings:
        return Settings(_env_file=None)

    def _enter(self, client):
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def login(self, customer_id: str = "usr_002") -> None:
        self.client.cookies.clear()
        response = self.client.post("/api/session", json={"customer_id": customer_id})
        self.assertEqual(response.status_code, 200, response.text)


class BootstrapTests(ApiTestCase):
    def test_bootstrap_returns_mode_roster_and_examples(self):
        data = self.client.get("/api/bootstrap").json()

        self.assertEqual(data["mode"], "scripted")
        self.assertIn("usr_002", {user["id"] for user in data["roster"]})
        self.assertEqual([e["key"] for e in data["examples"]], ["A", "B", "C", "D"])
        self.assertIsNone(data["me"])

    def test_index_serves_the_ui(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("SecureSupport", response.text)
        self.assertIn("Evidence &amp; trace", response.text)

    def test_health_reports_mode_and_no_secret(self):
        data = self.client.get("/api/health").json()

        self.assertEqual(data["mode"], "scripted")
        self.assertNotIn("openai_api_key", json.dumps(data))


class IdentityTests(ApiTestCase):
    def test_chat_without_a_session_is_rejected(self):
        response = self.client.post("/api/chat", json={"message": PAYMENT_QUESTION})

        self.assertEqual(response.status_code, 401)

    def test_unknown_customer_cannot_start_a_session(self):
        response = self.client.post("/api/session", json={"customer_id": "usr_999"})

        self.assertEqual(response.status_code, 400)

    def test_forged_cookie_is_rejected(self):
        self.client.cookies.set(COOKIE_NAME, "usr_001.deadbeef")

        response = self.client.post("/api/chat", json={"message": "What is the refund policy?"})

        self.assertEqual(response.status_code, 401)

    def test_session_cookie_is_signed_and_httponly(self):
        self.client.cookies.clear()
        response = self.client.post("/api/session", json={"customer_id": "usr_002"})
        cookie = response.headers["set-cookie"]

        self.assertIn("HttpOnly", cookie)
        self.assertNotIn("usr_002;", cookie)  # never a bare, forgeable id
        signer = DemoIdentitySigner(b"shared-secret")
        self.assertEqual(signer.verify(signer.sign("usr_002")), "usr_002")
        self.assertIsNone(signer.verify("usr_002.0000"))

    def test_customer_id_in_the_body_cannot_override_the_session(self):
        self.login("usr_002")

        # usr_001 does not own pay_002; if the body were trusted this would be refused.
        response = self.client.post("/api/chat", json={
            "message": PAYMENT_QUESTION,
            "customer_id": "usr_001",
        })

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["response"]["status"], "answered")
        self.assertEqual(data["response"]["tool_calls"][0]["outcome"]["code"], "ok")


class ScenarioTests(ApiTestCase):
    def test_payment_question_is_answered_with_evidence_and_trace(self):
        self.login("usr_002")

        response = self.client.post("/api/chat", json={"message": PAYMENT_QUESTION})

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        result, trace = data["response"], data["trace"]
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["source_ids"], ["kb_002"])
        self.assertEqual(result["tool_calls"][0]["tool_name"], "get_payment_status")
        self.assertEqual(result["tool_calls"][0]["outcome"]["code"], "ok")
        self.assertEqual(trace["trace_id"], result["trace_id"])
        self.assertEqual(trace["retrieval_hit"][0]["article_id"], "kb_002")
        self.assertGreaterEqual(trace["llm_calls"], 1)

    def test_cross_customer_request_is_refused_without_tools(self):
        self.login("usr_001")

        data = self.client.post("/api/chat", json={"message": CROSS_CUSTOMER}).json()

        self.assertEqual(data["response"]["status"], "refused")
        self.assertEqual(data["response"]["tool_calls"], [])
        self.assertEqual(data["response"]["source_ids"], [])

    def test_human_request_creates_a_real_ticket_id(self):
        self.login("usr_002")

        data = self.client.post("/api/chat", json={"message": HUMAN_REQUEST}).json()

        self.assertEqual(data["response"]["status"], "escalated")
        self.assertTrue(data["response"]["escalation_ticket_id"].startswith("tkt_"))
        self.assertEqual(data["response"]["escalation"]["reason"], "customer_requested")

    def test_reset_requires_post_and_clears_the_conversation(self):
        self.login("usr_002")

        # The UI calls this with POST; a GET must not silently appear to succeed.
        self.assertEqual(self.client.get("/api/reset").status_code, 405)
        self.assertEqual(self.client.post("/api/reset").status_code, 200)

    def test_unknown_message_in_scripted_mode_is_refused_not_invented(self):
        self.login("usr_002")

        response = self.client.post("/api/chat", json={"message": "Tell me a joke about cats."})

        self.assertEqual(response.status_code, 400)
        self.assertIn("preset questions", response.json()["detail"])

    def test_trace_does_not_contain_the_message_text(self):
        self.login("usr_002")

        data = self.client.post("/api/chat", json={"message": PAYMENT_QUESTION}).json()

        self.assertNotIn(PAYMENT_QUESTION.lower(), json.dumps(data["trace"]).lower())
        self.assertEqual(len(data["trace"]["message_fingerprint"]), 12)


class LiveModeWithoutCredentialsTests(ApiTestCase):
    mode = "live"

    def settings(self) -> Settings:
        return Settings(_env_file=None, openai_api_key=None)

    def test_live_mode_without_a_key_returns_actionable_setup_error(self):
        self.login("usr_002")

        response = self.client.post("/api/chat", json={"message": PAYMENT_QUESTION})

        self.assertEqual(response.status_code, 503)
        self.assertIn("SECURESUPPORT_OPENAI_API_KEY", response.json()["detail"])


class ConversationStoreTests(unittest.TestCase):
    def test_history_is_bounded_and_isolated_per_customer(self):
        store = ConversationStore(max_turns=4)
        for index in range(3):
            store.append("usr_002", f"question {index}", f"answer {index}")

        turns = store.history("usr_002")
        self.assertEqual(len(turns), 4)  # two exchanges, not three
        self.assertEqual(turns[0], ConversationTurn(role="customer", content="question 1"))
        self.assertEqual(store.history("usr_001"), [])

        store.reset("usr_002")
        self.assertEqual(store.history("usr_002"), [])

    def test_zero_budget_stores_nothing(self):
        store = ConversationStore(max_turns=0)
        store.append("usr_002", "hello", "hi")

        self.assertEqual(store.history("usr_002"), [])


if __name__ == "__main__":
    unittest.main()
