import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(dotenv_path=REPO_ROOT / ".env")

from agente_ai import config, llm_client
from agente_ai.graph.build_graph import graph

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.jsonl"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

JUDGE_SYSTEM_INSTRUCTION = """Eres un evaluador de calidad para un asistente de
medicina funcional. Compara la respuesta dada contra el contexto recuperado y
califica:

- faithfulness (1-5): cada afirmacion relevante de la respuesta esta respaldada
  por el contexto. 5 = totalmente respaldada, 1 = contradice o inventa datos.
- relevance (1-5): la respuesta atiende la pregunta del usuario. 5 = totalmente
  relevante, 1 = no responde la pregunta.

Responde siempre con el JSON solicitado."""


class JudgeScore(BaseModel):
    faithfulness: int
    relevance: int
    reasoning: str


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_answer(question: str, answer: str, chunks: list[dict]) -> JudgeScore:
    context = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks) or "(sin contexto)"
    prompt = f"Pregunta: {question}\n\nContexto disponible:\n{context}\n\nRespuesta a evaluar:\n{answer}"
    return llm_client.generate_structured(
        prompt, response_schema=JudgeScore, system_instruction=JUDGE_SYSTEM_INSTRUCTION
    )


def evaluate_case(case: dict) -> dict:
    result = graph.invoke({"query": case["question"]})
    entry: dict = {"question": case["question"], "route": result.get("route")}

    expected_route = case.get("expected_route")
    if expected_route:
        entry["route_correct"] = result.get("route") == expected_route

    expected_sources = case.get("expected_sources")
    if expected_sources:
        retrieved_docs = {c["doc_id"] for c in result.get("retrieved_chunks", [])}
        entry["retrieval_hit"] = bool(retrieved_docs & set(expected_sources))

    expect_tool = case.get("expect_tool")
    if expect_tool:
        entry["tool_used_correct"] = expect_tool in (result.get("tool_calls_used") or [])

    if result.get("final_answer") and result.get("route") != "reject":
        judge = judge_answer(case["question"], result["final_answer"], result.get("retrieved_chunks", []))
        entry["faithfulness"] = judge.faithfulness
        entry["relevance"] = judge.relevance
        entry["judge_reasoning"] = judge.reasoning

    return entry


def run() -> None:
    cases = load_golden_set()
    results = []
    for i, case in enumerate(cases):
        if i > 0:
            # Stay under the model's per-minute free-tier quota (see
            # config.EVAL_PACING_SECONDS) rather than relying on retries alone.
            time.sleep(config.EVAL_PACING_SECONDS)
        results.append(evaluate_case(case))

    faithfulness_scores = [r["faithfulness"] for r in results if "faithfulness" in r]
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0

    route_checks = [r["route_correct"] for r in results if "route_correct" in r]
    route_accuracy = sum(route_checks) / len(route_checks) if route_checks else 0.0

    summary = {
        "results": results,
        "avg_faithfulness": round(avg_faithfulness, 2),
        "route_accuracy": round(route_accuracy, 2),
        "faithfulness_threshold": config.FAITHFULNESS_THRESHOLD,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Route accuracy: {route_accuracy:.0%} ({sum(route_checks)}/{len(route_checks)})")
    print(f"Avg faithfulness: {avg_faithfulness:.2f} / 5 (threshold {config.FAITHFULNESS_THRESHOLD})")

    if avg_faithfulness < config.FAITHFULNESS_THRESHOLD:
        print("FAILED: faithfulness below threshold")
        sys.exit(1)
    print("PASSED")


if __name__ == "__main__":
    run()
