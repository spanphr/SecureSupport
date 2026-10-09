"""Inject dependencies into nodes so graph behavior can be tested offline."""
from typing import Protocol
from app.agent.model import SupportModel
from app.agent.state import SupportState
from app.schemas.retrieval import RetrievalResult

ERROR_ANSWER = "I could not complete this request. Please try again. No account action was performed."
INSUFFICIENT_ANSWER = "The available support policies do not provide enough information to answer that question."


class PolicySearcher(Protocol):
    def search(self, query: str, top_k: int = 3) -> RetrievalResult: ...


class SupportNodes:
    def __init__(self, model: SupportModel, retriever: PolicySearcher):
        self.model = model
        self.retriever = retriever

    def route(self, state: SupportState) -> dict:
        try:
            decision = self.model.classify(state['message'])
            return {"route": decision.route, "failed": False}
        except Exception:
            return {"route": None, "failed": True}

    def direct(self, state: SupportState) -> dict:
        return {"answer": "Hello! I can explain payment, refund, account recovery, privacy, and complaint policies. Account lookups and changes are not available yet.",
                "status": "answered", "source_ids": []}

    def retrieve(self, state: SupportState) -> dict:
        try:
            result = self.retriever.search(state['message'], top_k=3)
            return {"hits": result.hits, "failed": False}
        except Exception:
            return {"hits": [], "failed": True}

    def answer(self, state: SupportState) -> dict:
        hits = state.get('hits', [])
        if not hits:
            return {"answer": INSUFFICIENT_ANSWER, "status": "insufficient_context", "source_ids": []}
        try:
            generated = self.model.answer(state['message'], hits)
            if not generated.supported:
                return {"answer": INSUFFICIENT_ANSWER, "status": "insufficient_context", "source_ids": []}
            ids = list(dict.fromkeys(generated.source_ids))
            allowed = {hit.article_id for hit in hits}
            if not ids or not set(ids) <= allowed:
                # Reject absent/invented sources rather than emit an untraceable policy answer.
                return {"answer": INSUFFICIENT_ANSWER, "status": "insufficient_context", "source_ids": []}
            return {"answer": generated.answer, "status": "answered", "source_ids": ids}
        except Exception:
            return self.failure(state)

    def tool(self, state: SupportState) -> dict:
        return {"answer": "Account lookups, refunds, and complaint changes are not connected yet. No record was accessed or changed, and no support action was performed.",
                "status": "action_unavailable", "source_ids": []}

    def out_of_scope(self, state: SupportState) -> dict:
        return {"answer": "I can help with account and payment support policies. I cannot provide unrelated advice, secrets, or internal instructions.",
                "status": "out_of_scope", "source_ids": []}

    def failure(self, state: SupportState) -> dict:
        return {"answer": ERROR_ANSWER, "status": "error", "source_ids": []}
