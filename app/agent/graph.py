"""Finite single-turn graph, with explicit conditional branches and no tool execution."""
from langgraph.graph import END, START, StateGraph
from app.agent.model import SupportModel
from app.agent.nodes import PolicySearcher, SupportNodes
from app.agent.state import SupportState
from app.schemas.support import SupportRequest, SupportResponse


def build_graph(model: SupportModel, retriever: PolicySearcher):
    nodes = SupportNodes(model, retriever)
    graph = StateGraph(SupportState)
    for name in ('route', 'direct', 'retrieve', 'answer', 'tool', 'out_of_scope', 'failure'):
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, 'route')
    graph.add_conditional_edges(
        'route', lambda state: 'failure' if state.get('failed') else state['route'],
        {'direct': 'direct', 'retrieval': 'retrieve', 'tool': 'tool',
         'out_of_scope': 'out_of_scope', 'failure': 'failure'},
    )
    graph.add_conditional_edges(
        'retrieve', lambda state: 'failure' if state.get('failed') else 'answer',
        {'failure': 'failure', 'answer': 'answer'},
    )
    for name in ('direct', 'answer', 'tool', 'out_of_scope', 'failure'):
        graph.add_edge(name, END)
    return graph.compile()


class SupportAgent:
    def __init__(self, model: SupportModel, retriever: PolicySearcher):
        self.graph = build_graph(model, retriever)

    def run(self, message: str) -> SupportResponse:
        request = SupportRequest(message=message)
        # Start fresh for every request; multi-turn checkpoints are a later stage.
        state = self.graph.invoke({"message": request.message, "route": None,
                                   "hits": [], "source_ids": [], "failed": False})
        return SupportResponse(route=state['route'], status=state['status'],
                               answer=state['answer'], source_ids=state['source_ids'])
