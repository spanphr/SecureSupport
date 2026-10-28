"""Deterministic lexical retriever for offline tests and evaluations.

This is not the production retriever (that is Chroma + OpenAI embeddings). It is a
reproducible stand-in so pipeline behaviour and ranking metrics can be measured
without paid API calls, and so ranking mistakes - including a wrong article ranked
first - can be reproduced on demand.
"""

import math
import re
from collections import Counter

from app.schemas.domain import KnowledgeArticle
from app.schemas.retrieval import RetrievalHit, RetrievalResult

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from has have how i if in
    is it its may me my of on or our should so than that the their them then there these
    they this to us was we were what when where which who why will with would you your""".split()
)
_K1 = 1.5
_B = 0.75


def tokens(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


class LexicalRetriever:
    """BM25-style scoring over the real knowledge base; no network, no API key."""

    def __init__(self, articles: list[KnowledgeArticle]):
        if not articles:
            raise ValueError("At least one article is required.")
        self.articles = sorted(articles, key=lambda article: article.id)
        self._counts = {
            article.id: Counter(tokens(f"{article.title} {article.content} {' '.join(article.tags)}"))
            for article in self.articles
        }
        self._length = {key: sum(counts.values()) for key, counts in self._counts.items()}
        self._average = sum(self._length.values()) / len(self._length)
        self._document_frequency = Counter(
            token for counts in self._counts.values() for token in counts
        )
        self._by_id = {article.id: article for article in self.articles}

    def _score(self, query: list[str], article_id: str) -> float:
        counts = self._counts[article_id]
        total = 0.0
        for token in query:
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            documents = self._document_frequency[token]
            idf = math.log(1 + (len(self._counts) - documents + 0.5) / (documents + 0.5))
            norm = _K1 * (1 - _B + _B * self._length[article_id] / self._average)
            total += idf * (frequency * (_K1 + 1)) / (frequency + norm)
        return total

    def search(self, query: str, top_k: int = 3) -> RetrievalResult:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer between 1 and 10.")
        text = query.strip()
        if not text:
            raise ValueError("Query must not be empty.")
        query_tokens = tokens(text)
        scored = [(self._score(query_tokens, article.id), article.id) for article in self.articles]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        hits = []
        for score, article_id in scored[:top_k]:
            article = self._by_id[article_id]
            # Bounded, monotone distance: 1.0 means no lexical overlap at all.
            distance = 1.0 / (1.0 + score) if score > 0 else 1.0
            hits.append(RetrievalHit(article_id=article.id, title=article.title,
                                     content=article.content, distance=round(distance, 6)))
        return RetrievalResult(query=text, hits=hits)
