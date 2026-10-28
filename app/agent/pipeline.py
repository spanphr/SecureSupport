"""Single entry point for one support turn.

Trust boundary: `session` is supplied by the application and is the only source of
customer identity. Nothing in the request message or in model output can change it.
"""

from app.agent.graph import build_graph
from app.agent.model import SupportModel
from app.agent.nodes import ERROR_ANSWER, PipelineNodes, PolicySearcher
from app.config import Settings
from app.observability import (
    JsonLogTraceSink,
    NullTraceSink,
    TraceBuilder,
    TraceSink,
)
from app.schemas.pipeline import PipelineRequest, PipelineResponse, SessionContext
from app.tools.registry import WRITE_TOOL_NAMES, ToolExecutor
from app.tools.support import SupportTools


class SupportPipeline:
    def __init__(
        self,
        *,
        model: SupportModel,
        searcher: PolicySearcher,
        tools: SupportTools,
        settings: Settings,
        sink: TraceSink | None = None,
    ):
        self.settings = settings
        if sink is not None:
            self.sink: TraceSink = sink
        elif settings.trace_log_enabled:
            self.sink = JsonLogTraceSink()
        else:
            self.sink = NullTraceSink()
        self.graph = build_graph(PipelineNodes(model, searcher, ToolExecutor(tools), settings))

    def run(self, request: PipelineRequest, session: SessionContext) -> PipelineResponse:
        limit = self.settings.max_history_turns
        history = list(request.history[-limit:]) if limit else []
        trace = TraceBuilder(
            request.message,
            input_price=self.settings.input_cost_per_million_usd,
            output_price=self.settings.output_cost_per_million_usd,
        )
        final = self.graph.invoke({
            "request": request,
            "session": session,
            "history": history,
            "trace": trace,
            "hits": [],
            "tool_calls": [],
            "source_ids": [],
            "failed": False,
            "next": "failure",
        })

        status = final.get("status", "error")
        record = trace.finish(status)
        self.sink.emit(record)

        calls = list(final.get("tool_calls", []))
        decision = final.get("decision")
        # Reported from tool outcomes, never from what the answer text claims.
        performed = any(
            call.outcome.ok and call.tool_name in WRITE_TOOL_NAMES for call in calls
        )
        return PipelineResponse(
            trace_id=record.trace_id,
            status=status,
            answer=final.get("answer") or ERROR_ANSWER,
            category=decision.category if decision else None,
            action=decision.action if decision else None,
            source_ids=final.get("source_ids", []),
            tool_calls=calls,
            action_performed=performed,
            escalation=final.get("escalation"),
            escalation_ticket_id=final.get("escalation_ticket_id"),
            total_latency_ms=record.total_latency_ms,
        )
