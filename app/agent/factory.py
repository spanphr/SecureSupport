"""Create live dependencies, deferring index access until a policy search is needed."""
from app.agent.graph import SupportAgent
from app.agent.model import OpenAISupportModel
from app.config import Settings
from app.schemas.retrieval import RetrievalResult


class LazyPolicySearcher:
    def __init__(self):
        self.retriever = None

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        if self.retriever is None:
            from app.retrieval.factory import build_retriever
            self.retriever = build_retriever()
        return self.retriever.search(query, top_k)


def build_agent() -> SupportAgent:
    return SupportAgent(OpenAISupportModel(Settings()), LazyPolicySearcher())
