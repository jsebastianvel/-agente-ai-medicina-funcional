MAX_CONTEXT_CHARS = 6000

ROUTER_SYSTEM_INSTRUCTION = """Eres el enrutador de un asistente de medicina funcional.
Clasifica la consulta del usuario en una de estas rutas:

- "rag_qa": preguntas de conocimiento general sobre medicina funcional, nutricion,
  microbiota, inflamacion, estres, detoxificacion, etc.
- "symptom_check": el usuario describe sintomas propios (fatiga, dolor, problemas
  digestivos, etc.) y busca entender a que podrian deberse.
- "reject": la consulta no tiene relacion alguna con salud, nutricion o medicina
  funcional (por ejemplo, geografia, matematicas, programacion, etc.).

Responde siempre con el JSON solicitado."""


def build_router_prompt(query: str) -> str:
    return f"Consulta del usuario: {query}"


def _format_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Shared by the drafting and critic prompts. Caps total context length as
    a guardrail against pathological cases (duplicate/oversized chunks) rather
    than trusting retrieval to always return a small, bounded result."""
    if not chunks:
        return "No hay informacion relevante en la base de conocimiento."
    parts: list[str] = []
    total = 0
    for c in chunks:
        block = f"[Fuente: {c['title']}]\n{c['text']}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


DRAFT_SYSTEM_INSTRUCTION = """Eres un asistente informativo de medicina funcional.
Respondes en espanol, de forma clara y breve, basandote unicamente en el
contexto proporcionado. No inventes datos que no esten en el contexto.
No reemplazas el consejo de un profesional de la salud: si es relevante,
sugiere consultar a un especialista."""


def build_draft_prompt(
    query: str,
    chunks: list[dict],
    symptom_match: dict | None = None,
    critic_feedback: str | None = None,
) -> str:
    parts = [f"Contexto recuperado:\n{_format_context(chunks)}"]

    if symptom_match:
        parts.append(
            f"Nota del chequeo de sintomas: {symptom_match['note']} "
            f"(eje relacionado: {symptom_match['axis']})"
        )

    if critic_feedback:
        parts.append(
            "Tu respuesta anterior fue revisada y necesita mejorar por esto: "
            f"{critic_feedback}\nCorrige eso en la nueva respuesta."
        )

    parts.append(f"Pregunta del usuario: {query}")
    parts.append("Responde a la pregunta citando el titulo de la fuente que respalda cada afirmacion.")
    return "\n\n".join(parts)


CRITIC_SYSTEM_INSTRUCTION = """Eres el revisor de calidad de un asistente de medicina
funcional. Tu trabajo es verificar que la respuesta propuesta este respaldada
por el contexto recuperado, sin inventar datos.

- "approved": toda afirmacion relevante de la respuesta esta respaldada por el contexto.
- "revise": la respuesta tiene afirmaciones no respaldadas o se puede mejorar
  citando mejor el contexto disponible; explica que corregir en "feedback".
- "insufficient_evidence": el contexto recuperado simplemente no alcanza para
  responder la pregunta con honestidad, sin importar como se redacte.

Responde siempre con el JSON solicitado."""


def build_critic_prompt(query: str, draft_answer: str, chunks: list[dict]) -> str:
    return (
        f"Pregunta del usuario: {query}\n\n"
        f"Contexto recuperado:\n{_format_context(chunks)}\n\n"
        f"Respuesta propuesta:\n{draft_answer}"
    )
