MIN_QUERY_CHARS = 3


def validate_query(query: str) -> tuple[bool, str | None]:
    """Deterministic, cheap pre-filter — the router LLM call handles topical
    classification, so this only rejects input that's clearly not a query
    at all (empty, whitespace, or a couple of stray characters)."""
    stripped = query.strip()
    if not stripped:
        return False, "La consulta esta vacia."
    if len(stripped) < MIN_QUERY_CHARS:
        return False, "La consulta es demasiado corta."
    return True, None
