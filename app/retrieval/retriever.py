"""Persistent cosine search over knowledge articles only."""
import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.retrieval.embeddings import Embedder
from app.schemas.domain import KnowledgeArticle
from app.schemas.retrieval import RetrievalHit, RetrievalResult


class KnowledgeRetriever:
    def __init__(self, articles: list[KnowledgeArticle], embedder: Embedder, path: Path):
        if not articles or len({a.id for a in articles}) != len(articles):
            raise ValueError("Knowledge articles must be nonempty with unique IDs.")
        self.articles = sorted(articles, key=lambda article: article.id)
        self.embedder = embedder
        # A changed corpus or embedding model gets its own collection, avoiding stale records.
        manifest = {"format": 1, "embedder": embedder.identity,
                    "articles": [a.model_dump() for a in self.articles]}
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=f"support_kb_{fingerprint}", embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    @property
    def ready(self) -> bool:
        return self.collection.count() == len(self.articles)

    def index(self) -> int:
        if self.ready:
            return self.collection.count()
        documents = [f"{a.title}\n{a.content}\nTags: {', '.join(a.tags)}" for a in self.articles]
        vectors = self.embedder.embed(documents)
        if len(vectors) != len(documents):
            raise ValueError("Embedding provider returned an incorrect vector count.")
        self.collection.upsert(
            ids=[a.id for a in self.articles], documents=documents, embeddings=vectors,
            metadatas=[{"title": a.title, "content": a.content} for a in self.articles],
        )
        return self.collection.count()

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        query = query.strip()
        if not query or len(query) > 4000:
            raise ValueError("Query must contain between 1 and 4000 characters.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer between 1 and 10.")
        if not self.ready:
            raise ValueError("Knowledge index is missing; run scripts.index_knowledge first.")
        vectors = self.embedder.embed([query])
        if len(vectors) != 1:
            raise ValueError("Embedding provider returned an incorrect query vector count.")
        result = self.collection.query(
            query_embeddings=vectors, n_results=min(top_k, self.collection.count()),
            include=["metadatas", "distances"],
        )
        hits = [RetrievalHit(article_id=identifier, title=metadata["title"],
                             content=metadata["content"], distance=distance)
                for identifier, metadata, distance in zip(
                    result["ids"][0], result["metadatas"][0], result["distances"][0], strict=True)]
        return RetrievalResult(query=query, hits=hits)
