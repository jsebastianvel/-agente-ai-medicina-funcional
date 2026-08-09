from agente_ai.guardrails.validators import validate_query


def test_validate_query_rejects_empty_string():
    valid, reason = validate_query("")

    assert valid is False
    assert reason


def test_validate_query_rejects_whitespace_only():
    valid, reason = validate_query("   ")

    assert valid is False
    assert reason


def test_validate_query_rejects_too_short():
    valid, reason = validate_query("hi")

    assert valid is False
    assert reason


def test_validate_query_accepts_normal_question():
    valid, reason = validate_query("¿Que es la medicina funcional?")

    assert valid is True
    assert reason is None
