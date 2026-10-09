from app.retrieval.factory import build_retriever


def main():
    try:
        retriever = build_retriever()
        print(f"Knowledge index ready: {retriever.index()} articles.")
    except Exception:
        # Do not print SDK exceptions, request bodies, or configuration values.
        raise SystemExit("Indexing failed. Check local API-key configuration, network access, and index permissions.") from None


if __name__ == "__main__":
    main()
