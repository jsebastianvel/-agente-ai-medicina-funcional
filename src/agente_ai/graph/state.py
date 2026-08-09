from typing import Literal, TypedDict


class RetrievedChunk(TypedDict):
    text: str
    doc_id: str
    title: str
    source_url: str


class GraphState(TypedDict, total=False):
    query: str
    query_valid: bool
    rejection_reason: str | None

    route: Literal["rag_qa", "symptom_check", "reject"]
    tool_calls_used: list[str]
    symptom_match: dict | None

    retrieved_chunks: list[RetrievedChunk]

    draft_answer: str
    critic_verdict: Literal["approved", "revise", "insufficient_evidence"]
    critic_feedback: str
    revision_count: int

    final_answer: str
    citations: list[str]
