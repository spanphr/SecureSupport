"""Create live dependencies, deferring index access until a policy search is needed."""

from app.agent.model import OpenAISupportModel
from app.agent.pipeline import SupportPipeline
from app.config import Settings
from app.observability import TraceSink
from app.schemas.retrieval import RetrievalResult
from app.storage.database import Database
from app.storage.repository import SupportRepository
from app.tools.support import SupportTools


class LazyPolicySearcher:
    """Avoids loading Chroma and creating an embedding client until retrieval is used."""

    def __init__(self):
        self.retriever = None

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        if self.retriever is None:
            from app.retrieval.factory import build_retriever
            self.retriever = build_retriever()
        return self.retriever.search(query, top_k)


def build_pipeline(settings: Settings | None = None, sink: TraceSink | None = None) -> SupportPipeline:
    settings = settings or Settings()
    tools = SupportTools(SupportRepository(Database(settings.database_path)))
    return SupportPipeline(
        model=OpenAISupportModel(settings),
        searcher=LazyPolicySearcher(),
        tools=tools,
        settings=settings,
        sink=sink,
    )
