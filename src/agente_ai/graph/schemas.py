from typing import Literal

from pydantic import BaseModel


class RouteDecision(BaseModel):
    route: Literal["rag_qa", "symptom_check", "reject"]
    reasoning: str


class CriticVerdict(BaseModel):
    verdict: Literal["approved", "revise", "insufficient_evidence"]
    feedback: str
