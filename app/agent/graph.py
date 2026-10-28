"""Bounded workflow graph.

This is a workflow, not an autonomous agent: the path is fixed, the loop is
impossible, and every branch terminates. Each non-terminal node sets `next`, which
is the only routing signal - there is no free-form planner step.
"""

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import PipelineNodes
from app.agent.state import PipelineState

NODE_NAMES = (
    "validate", "decide", "respond", "clarify", "refuse", "retrieve",
    "compose", "tool", "escalate", "check_output", "failure",
)
# Every branching node must set `next` on every return path. Terminal nodes have a
# plain edge to END; they never consult `next`, so no stale value can create a cycle.
BRANCHING = ("validate", "decide", "retrieve", "compose", "tool", "escalate")
TERMINAL = ("respond", "clarify", "refuse", "check_output", "failure")


def _route(state: PipelineState) -> str:
    return state["next"] if "next" in state else "failure"


def build_graph(nodes: PipelineNodes):
    graph = StateGraph(PipelineState)
    for name in NODE_NAMES:
        graph.add_node(name, getattr(nodes, name))

    targets = {name: name for name in NODE_NAMES}
    graph.add_edge(START, "validate")
    for name in BRANCHING:
        graph.add_conditional_edges(name, _route, targets)
    for name in TERMINAL:
        graph.add_edge(name, END)
    return graph.compile()
