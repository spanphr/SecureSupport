import argparse
from app.retrieval.factory import build_retriever


def main():
    parser = argparse.ArgumentParser(description="Search synthetic support policies.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 11))
    args = parser.parse_args()
    try:
        result = build_retriever().search(args.query, args.top_k)
    except Exception:
        raise SystemExit("Search failed. Check API-key configuration and run scripts.index_knowledge first.") from None
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
