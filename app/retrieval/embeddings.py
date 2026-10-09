"""Explicit embeddings provider; importing this module makes no API calls."""
from typing import Protocol
from openai import OpenAI
from app.config import Settings


class Embedder(Protocol):
    identity: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, settings: Settings):
        key = settings.openai_api_key
        if key is None or not key.get_secret_value().strip():
            raise ValueError("Set SECURESUPPORT_OPENAI_API_KEY in your local .env before live retrieval.")
        self.model = settings.embedding_model
        self.identity = f"openai:{self.model}:default-dimensions"
        self.client = OpenAI(api_key=key.get_secret_value(), timeout=30.0, max_retries=2)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model, input=texts, encoding_format="float"
        )
        items = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(texts))):
            raise ValueError("Embedding response did not match requested inputs.")
        return [item.embedding for item in items]
