from agente_ai import config, llm_client
from agente_ai.graph.prompts import (
    CRITIC_SYSTEM_INSTRUCTION,
    DRAFT_SYSTEM_INSTRUCTION,
    ROUTER_SYSTEM_INSTRUCTION,
    build_critic_prompt,
    build_draft_prompt,
    build_router_prompt,
)
from agente_ai.graph.schemas import CriticVerdict, RouteDecision
from agente_ai.graph.state import GraphState
from agente_ai.guardrails.validators import validate_query
from agente_ai.observability.logger import log_node
from agente_ai.rag.retriever import retrieve as rag_retrieve
from agente_ai.tools.symptom_checker_tool import check_symptoms

REJECTION_MESSAGE = (
    "Esa consulta no parece estar relacionada con medicina funcional, nutricion "
    "o bienestar. Puedo ayudarte con preguntas sobre esos temas."
)
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "No tengo suficiente informacion en mi base de conocimiento para responder "
    "esto con confianza. Prefiero decirtelo antes que inventar una respuesta."
)


@log_node("validate_input")
def validate_input(state: GraphState) -> dict:
    valid, reason = validate_query(state["query"])
    return {"query_valid": valid, "rejection_reason": reason}


@log_node("router")
def router(state: GraphState) -> dict:
    decision: RouteDecision = llm_client.generate_structured(
        build_router_prompt(state["query"]),
        response_schema=RouteDecision,
        system_instruction=ROUTER_SYSTEM_INSTRUCTION,
    )
    return {"route": decision.route}


@log_node("symptom_check")
def symptom_check(state: GraphState) -> dict:
    match = check_symptoms(state["query"])
    tool_calls = state.get("tool_calls_used", []) + ["symptom_checker_tool"]
    return {"symptom_match": match, "tool_calls_used": tool_calls}


@log_node("retrieve")
def retrieve(state: GraphState) -> dict:
    chunks = rag_retrieve(state["query"])
    tool_calls = state.get("tool_calls_used", []) + ["retrieval_tool"]
    return {"retrieved_chunks": chunks, "tool_calls_used": tool_calls}


@log_node("draft_answer")
def draft_answer(state: GraphState) -> dict:
    prompt = build_draft_prompt(
        state["query"],
        state.get("retrieved_chunks", []),
        symptom_match=state.get("symptom_match"),
        critic_feedback=state.get("critic_feedback") if state.get("revision_count") else None,
    )
    answer = llm_client.generate_text(prompt, system_instruction=DRAFT_SYSTEM_INSTRUCTION)
    return {"draft_answer": answer}


@log_node("critic")
def critic(state: GraphState) -> dict:
    verdict: CriticVerdict = llm_client.generate_structured(
        build_critic_prompt(state["query"], state["draft_answer"], state.get("retrieved_chunks", [])),
        response_schema=CriticVerdict,
        system_instruction=CRITIC_SYSTEM_INSTRUCTION,
    )
    return {
        "critic_verdict": verdict.verdict,
        "critic_feedback": verdict.feedback,
        "revision_count": state.get("revision_count", 0) + 1,
    }


@log_node("format_final")
def format_final(state: GraphState) -> dict:
    citations = sorted({c["title"] for c in state.get("retrieved_chunks", [])})
    return {"final_answer": state["draft_answer"], "citations": citations}


@log_node("reject_response")
def reject_response(state: GraphState) -> dict:
    return {"final_answer": REJECTION_MESSAGE, "citations": []}


@log_node("insufficient_evidence_response")
def insufficient_evidence_response(state: GraphState) -> dict:
    return {"final_answer": INSUFFICIENT_EVIDENCE_MESSAGE, "citations": []}


def route_after_validate(state: GraphState) -> str:
    return "router" if state["query_valid"] else "reject_response"


def route_after_router(state: GraphState) -> str:
    return state["route"]


def route_after_critic(state: GraphState) -> str:
    if state["critic_verdict"] == "approved":
        return "approved"
    if state["revision_count"] >= config.MAX_REVISIONS:
        return "insufficient_evidence"
    return "revise"
