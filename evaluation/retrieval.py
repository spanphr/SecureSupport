"""Small labeled smoke evaluation; MRR is truncated at retrieved top three."""
import json
from pathlib import Path
from typing import Protocol
from pydantic import Field
from app.schemas.domain import Record
from app.schemas.retrieval import RetrievalResult

CASES_PATH = Path(__file__).resolve().parent / "retrieval_cases.json"


class RetrievalCase(Record):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_article_ids: list[str] = Field(min_length=1)


class Searcher(Protocol):
    def search(self, query: str, top_k: int = 3) -> RetrievalResult: ...


def evaluate(retriever: Searcher, cases: list[RetrievalCase]) -> dict:
    if not cases:
        raise ValueError("Evaluation needs at least one case.")
    rows = []
    for case in cases:
        ids = [hit.article_id for hit in retriever.search(case.query, top_k=3).hits][:3]
        rank = next((i for i, identifier in enumerate(ids, 1)
                     if identifier in case.relevant_article_ids), None)
        rows.append({"case_id": case.id, "retrieved_ids": ids,
                     "hit_at_1": int(rank == 1), "hit_at_3": int(rank is not None),
                     "reciprocal_rank": 1 / rank if rank else 0.0})
    return {"cases": len(rows), "hit_at_1": sum(r['hit_at_1'] for r in rows) / len(rows),
            "hit_at_3": sum(r['hit_at_3'] for r in rows) / len(rows),
            "mrr_at_3": sum(r['reciprocal_rank'] for r in rows) / len(rows), "results": rows}


def load_cases() -> list[RetrievalCase]:
    return [RetrievalCase.model_validate(case) for case in json.loads(CASES_PATH.read_text())]


def main():
    from app.retrieval.factory import build_retriever
    try:
        retriever = build_retriever()
        cases = load_cases()
        article_ids = {article.id for article in retriever.articles}
        if any(set(case.relevant_article_ids) - article_ids for case in cases):
            raise ValueError("Evaluation references unknown articles.")
        report = evaluate(retriever, cases)
    except Exception:
        raise SystemExit("Evaluation failed. Check API-key configuration and knowledge index readiness.") from None
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
