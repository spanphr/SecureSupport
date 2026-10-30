"""HTTP transport for the support pipeline.

A deliberately thin adapter. It resolves identity, keeps bounded conversation history,
and serializes results; every guardrail, decision, tool call, and escalation stays in
`app.agent`. Nothing in this module can widen what the pipeline allows.

Two modes:

* ``scripted`` (default for the CLI): canned model output, real everything else. Free,
  deterministic, and the only mode the test suite exercises.
* ``live``: the configured OpenAI model and Chroma index, spending tokens on your key.

Run with ``python -m scripts.serve`` (``--live`` for the real model).
"""

import logging
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent.factory import build_pipeline
from app.agent.pipeline import SupportPipeline
from app.api.demo import ScriptedDemo
from app.api.identity import COOKIE_NAME, DemoIdentitySigner
from app.config import Settings
from app.demo_scenarios import SCENARIOS
from app.observability import JsonLogTraceSink, TraceRecord, TraceSink
from app.schemas.pipeline import ConversationTurn, PipelineRequest, SessionContext
from scripts.validate_data import validate_dataset

LOGGER = logging.getLogger("securesupport.api")
INDEX_HTML = Path(__file__).resolve().parents[2] / "frontend" / "index.html"
COOKIE_MAX_AGE = 8 * 60 * 60

LIVE_SETUP_MESSAGE = (
    "Live mode needs SECURESUPPORT_OPENAI_API_KEY in .env and a built retrieval index "
    "(python -m scripts.index_knowledge). Start the server without --live for the free "
    "scripted demo."
)
SCRIPTED_HINT = (
    "The scripted demo only answers its preset questions. Pick one of the examples, or "
    "restart the server with --live to use the real model."
)


class TraceBuffer:
    """Keeps recent trace records so a response can be shown with its own trace."""

    def __init__(self, limit: int = 100):
        self._records: deque[TraceRecord] = deque(maxlen=limit)
        self._lock = threading.Lock()

    def emit(self, record: TraceRecord) -> None:
        with self._lock:
            self._records.append(record)

    def get(self, trace_id: str) -> TraceRecord | None:
        with self._lock:
            for record in reversed(self._records):
                if record.trace_id == trace_id:
                    return record
        return None


class TeeTraceSink:
    """Fan a trace out to the structured log and the response buffer."""

    def __init__(self, *sinks: TraceSink):
        self._sinks = sinks

    def emit(self, record: TraceRecord) -> None:
        for sink in self._sinks:
            sink.emit(record)


class ConversationStore:
    """Bounded, in-memory history per customer.

    Server-side on purpose: history supplied by a browser could contain forged agent
    turns. A restart drops it, which is acceptable for a demo and recorded as a
    limitation.
    """

    def __init__(self, max_turns: int):
        self._max = max(0, max_turns)
        self._turns: dict[str, list[ConversationTurn]] = {}
        self._lock = threading.Lock()

    def history(self, customer_id: str) -> list[ConversationTurn]:
        with self._lock:
            return list(self._turns.get(customer_id, []))

    def append(self, customer_id: str, message: str, answer: str) -> None:
        if not self._max:
            return
        with self._lock:
            turns = self._turns.setdefault(customer_id, [])
            turns.append(ConversationTurn(role="customer", content=message[:4000]))
            turns.append(ConversationTurn(role="agent", content=answer[:4000]))
            del turns[: max(0, len(turns) - self._max)]

    def reset(self, customer_id: str) -> None:
        with self._lock:
            self._turns.pop(customer_id, None)


class SessionBody(BaseModel):
    customer_id: str = Field(min_length=1, max_length=100)


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # Set by the application after an explicit customer confirmation; never inferred
    # from the message text.
    confirm_write: bool = False


def _roster() -> list[dict[str, str]]:
    """The synthetic demo customers. Names are fictional."""
    return [{"id": user.id, "name": user.name} for user in validate_dataset().users]


def create_app(*, mode: Literal["live", "scripted"] = "scripted",
               settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    signer = DemoIdentitySigner()
    buffer = TraceBuffer()
    store = ConversationStore(settings.max_history_turns)
    demo = ScriptedDemo(settings) if mode == "scripted" else None
    sinks: list[TraceSink] = [buffer]
    if settings.trace_log_enabled:
        sinks.append(JsonLogTraceSink())
    sink = TeeTraceSink(*sinks)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if demo is not None:
            demo.close()

    app = FastAPI(
        title="SecureSupport",
        version="1.0.0",
        description="Guardrailed agentic customer support: retrieval, authorized tools, "
                    "verification, escalation.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    holder: dict[str, SupportPipeline | None] = {"pipeline": None}
    lock = threading.Lock()

    def identity(request: Request) -> str:
        customer_id = signer.verify(request.cookies.get(COOKIE_NAME))
        if customer_id is None:
            raise HTTPException(status_code=401, detail="Choose a demo customer to start.")
        return customer_id

    def live_pipeline() -> SupportPipeline:
        with lock:
            if holder["pipeline"] is None:
                if settings.openai_api_key is None:
                    raise HTTPException(status_code=503, detail=LIVE_SETUP_MESSAGE)
                try:
                    holder["pipeline"] = build_pipeline(settings=settings, sink=sink)
                except Exception as error:  # class name only: messages can leak config
                    LOGGER.error("live-pipeline-startup-failed %s", type(error).__name__)
                    raise HTTPException(status_code=503, detail=LIVE_SETUP_MESSAGE) from None
            return holder["pipeline"]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        if not INDEX_HTML.exists():
            raise HTTPException(status_code=500, detail="frontend/index.html is missing.")
        return FileResponse(INDEX_HTML)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "mode": mode,
            "chat_model": settings.chat_model,
            "live_ready": settings.openai_api_key is not None,
            "retrieval_top_k": settings.retrieval_top_k,
        }

    @app.get("/api/bootstrap")
    def bootstrap(request: Request) -> dict:
        return {
            "mode": mode,
            "roster": _roster(),
            "me": signer.verify(request.cookies.get(COOKIE_NAME)),
            "examples": [
                {"key": s.key, "title": s.title, "message": s.message,
                 "customer_id": s.customer_id}
                for s in sorted(SCENARIOS.values(), key=lambda s: s.key)
            ],
        }

    @app.post("/api/session")
    def start_session(body: SessionBody, response: Response) -> dict:
        if body.customer_id not in {item["id"] for item in _roster()}:
            raise HTTPException(status_code=400, detail="Unknown demo customer.")
        store.reset(body.customer_id)
        response.set_cookie(
            COOKIE_NAME, signer.sign(body.customer_id), httponly=True, samesite="lax",
            max_age=COOKIE_MAX_AGE, path="/",
        )
        return {"customer_id": body.customer_id}

    @app.post("/api/reset")
    def reset(request: Request) -> dict:
        store.reset(identity(request))
        return {"ok": True}

    @app.post("/api/chat")
    def chat(body: ChatBody, request: Request) -> dict:
        customer_id = identity(request)
        message = body.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message must not be empty.")

        if demo is not None:
            scenario = demo.match(message)
            if scenario is None:
                raise HTTPException(status_code=400, detail=SCRIPTED_HINT)
            pipeline = demo.pipeline(scenario, sink)
        else:
            pipeline = live_pipeline()

        history = store.history(customer_id)
        try:
            result = pipeline.run(
                PipelineRequest(message=message, history=history,
                                confirm_write=body.confirm_write),
                SessionContext(customer_id=customer_id),
            )
        except Exception as error:
            LOGGER.error("pipeline-crash %s", type(error).__name__)
            raise HTTPException(
                status_code=500,
                detail="The request could not be completed. No action was performed.",
            ) from None

        store.append(customer_id, message, result.answer)
        trace = buffer.get(result.trace_id)
        return {
            "mode": mode,
            "response": result.model_dump(mode="json"),
            "trace": None if trace is None else trace.model_dump(mode="json"),
        }

    return app
