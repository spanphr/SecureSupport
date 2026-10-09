# SecureSupport

## A Guardrailed Agentic Customer Support System

Foundation and synthetic domain data, using Python 3.11+.
There is no business logic, agent workflow, retrieval implementation, API server,
or frontend application yet.

## Local setup

Run these commands from the SecureSupport project root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

No API key is required for the foundation. Do not commit real keys.
No API calls are made by this project.

## Configuration

`app/config.py` defines a Pydantic `Settings` class. Instantiate it when needed:

```python
from app.config import Settings

settings = Settings()
```

It reads `SECURESUPPORT_` environment variables and an optional `.env` file
relative to the current working directory. Environment variables override `.env`.
Defaults are development mode, debug disabled, and no OpenAI API key.
An optional key uses `SecretStr` to mask it in normal representations.

## Folder structure

- `app/`: Python package and configuration.
- `app/schemas/`: validated synthetic domain models.
- `app/agent/`: reserved for future agent workflows.
- `app/retrieval/`: reserved for future retrieval code.
- `app/tools/`: reserved for future support tools.
- `app/guardrails/`: reserved for future guardrails.
- `data/`: synthetic support fixtures.
- `evaluation/`: reserved for future evaluation code.
- `tests/`: fixture validation and invalid-input checks.
- `scripts/`: fixture validation utilities.
- `frontend/`: reserved for the future user interface.

## Planned future stack

OpenAI, LangGraph, Pydantic, Chroma, SQLite/SQLAlchemy, FastAPI, Streamlit,
Pytest, and Docker. Only Pydantic Settings is required at this stage; other
components are deferred.

See `DECISIONS.md` for foundation decisions. Run `git status` to inspect the
initial uncommitted files.

## Synthetic data

`data/synthetic_support.json` contains fictional users, payments, tickets,
knowledge articles, complaints, and six deliberate failure prompts. All policies
are demo policies, not production commitments. Emails use the reserved `.invalid`
domain; no credentials or real customer information are included. Amounts are
integer AUD cents. These fixtures are not an authentication mechanism.

After installing the requirements in your local virtual environment, run:

```sh
python -m scripts.validate_data
python -m unittest discover -s tests -v
```

Validation checks field constraints, duplicate IDs, foreign references, and
complaint ownership. Failure prompts are evaluation inputs with expected behavior;
they do not constitute an implemented or tested agent guardrail.

Next: build retrieval over the knowledge articles and a labeled retrieval
evaluation set before adding the agent workflow.
