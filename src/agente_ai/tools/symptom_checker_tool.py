"""A deterministic, zero-cost tool: no LLM call, no external API. Maps
self-reported symptom keywords to the functional-medicine "axis" the corpus
covers, so the drafting node has a concrete lead to search for even when the
user describes symptoms rather than asking a direct knowledge question."""

SYMPTOM_RULES = [
    {
        "keywords": ["fatiga", "cansancio", "estres", "estrés", "insomnio", "agotamiento"],
        "axis": "eje HPA / estrés / cortisol",
        "note": "Posible relación con el eje HPA (estrés/cortisol).",
    },
    {
        "keywords": ["hinchazon", "hinchazón", "gases", "digestion", "digestión",
                     "intestino irritable", "diarrea", "estreñimiento"],
        "axis": "microbiota intestinal",
        "note": "Posible relación con la microbiota intestinal.",
    },
    {
        "keywords": ["dolor articular", "rigidez", "hinchazon cronica", "hinchazón crónica"],
        "axis": "inflamación crónica de bajo grado",
        "note": "Posible relación con inflamación crónica de bajo grado.",
    },
    {
        "keywords": ["alergia", "intolerancia", "sensibilidad alimentaria"],
        "axis": "dieta de eliminación",
        "note": "Posible relación con sensibilidades alimentarias; una dieta de eliminación podría ayudar a identificarlas.",
    },
]


def check_symptoms(text: str) -> dict | None:
    """Pure keyword-match lookup, no LLM call. Returns the first matching
    rule's axis+note, or None if no keyword matches."""
    lowered = text.lower()
    for rule in SYMPTOM_RULES:
        if any(keyword in lowered for keyword in rule["keywords"]):
            return {"axis": rule["axis"], "note": rule["note"]}
    return None
