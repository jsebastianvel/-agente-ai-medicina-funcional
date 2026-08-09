from langgraph.graph import END, StateGraph

from agente_ai.graph.nodes import (
    critic,
    draft_answer,
    format_final,
    insufficient_evidence_response,
    reject_response,
    retrieve,
    route_after_critic,
    route_after_router,
    route_after_validate,
    router,
    symptom_check,
    validate_input,
)
from agente_ai.graph.state import GraphState


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("validate_input", validate_input)
    g.add_node("router", router)
    g.add_node("retrieve", retrieve)
    g.add_node("symptom_check", symptom_check)
    g.add_node("draft_answer", draft_answer)
    g.add_node("critic", critic)
    g.add_node("format_final", format_final)
    g.add_node("reject_response", reject_response)
    g.add_node("insufficient_evidence_response", insufficient_evidence_response)

    g.set_entry_point("validate_input")

    g.add_conditional_edges(
        "validate_input",
        route_after_validate,
        {"router": "router", "reject_response": "reject_response"},
    )
    g.add_conditional_edges(
        "router",
        route_after_router,
        {"rag_qa": "retrieve", "symptom_check": "symptom_check", "reject": "reject_response"},
    )
    g.add_edge("symptom_check", "retrieve")
    g.add_edge("retrieve", "draft_answer")
    g.add_edge("draft_answer", "critic")

    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "approved": "format_final",
            "revise": "draft_answer",
            "insufficient_evidence": "insufficient_evidence_response",
        },
    )
    g.add_edge("format_final", END)
    g.add_edge("reject_response", END)
    g.add_edge("insufficient_evidence_response", END)

    return g.compile()


graph = build_graph()
