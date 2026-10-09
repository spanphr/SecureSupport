# Foundation decisions

1. Use Python 3.11 or newer.
2. Keep this stage limited to project structure and configuration.
3. Use Pydantic Settings v2 for validated environment-based configuration.
   Its dependencies include Pydantic and dotenv support, so only
   `pydantic-settings` is listed as a direct dependency.
4. Prefix configuration variables with `SECURESUPPORT_`.
5. Make API credentials optional and mask their normal representation with
   `SecretStr`. Commit only a blank key placeholder in `.env.example`.
6. Define the settings class without a global instance; importing the module
   does not load configuration or contact external services.
7. Use `__init__.py` in Python package directories and `.gitkeep` in reserved
   non-package directories so the initial structure survives a Git checkout.
8. Defer business logic, LangGraph, Chroma, SQLite/SQLAlchemy integrations,
   FastAPI, Streamlit, evaluation logic, tests, and Docker setup to later stages.
9. Initialize a local Git repository without creating an initial commit.

## Synthetic domain data

10. Keep fixtures deterministic and fictional, with `.invalid` email addresses.
11. Store payment amounts as integer cents to avoid floating-point money errors.
12. Validate record schemas, unique IDs, user references, and complaint ownership.
13. Treat knowledge articles as demo policies and failure prompts as future
    evaluation cases. No agent, CRUD actions, or runtime guardrails are added yet.
14. Use standard-library unittest for data checks; no test dependency is needed.

## Knowledge retrieval

15. Use explicit OpenAI vectors and Chroma cosine distance; no implicit embedding
    function or model download. Default model: text-embedding-3-small.
16. Index knowledge articles only; return source IDs and text for later grounding.
17. Version collections by article content and embedding provider/model identity
    so changed policies and models cannot silently reuse an old index.
18. Report Hit@1, Hit@3, and MRR@3 on a 12-query demo set. Validate plumbing offline;
    report live semantic quality only after an actual API-backed evaluation.
19. Keep generated vectors and environments out of Git. Do not add answer
    generation, routing, or sensitive actions during this retrieval step.
