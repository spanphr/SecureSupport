"""Build the live retriever only when explicitly requested."""
from app.config import Settings
from app.retrieval.embeddings import OpenAIEmbedder
from app.retrieval.retriever import KnowledgeRetriever
from scripts.validate_data import validate_dataset


def build_retriever() -> KnowledgeRetriever:
    settings = Settings()
    embedder = OpenAIEmbedder(settings)
    return KnowledgeRetriever(validate_dataset().knowledge_base, embedder, settings.chroma_path)
