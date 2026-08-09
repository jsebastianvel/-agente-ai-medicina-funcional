from agente_ai.tools.symptom_checker_tool import check_symptoms


def test_check_symptoms_matches_fatigue_to_hpa_axis():
    result = check_symptoms("Tengo fatiga cronica e insomnio desde hace semanas")

    assert result is not None
    assert result["axis"] == "eje HPA / estrés / cortisol"


def test_check_symptoms_matches_digestive_keywords():
    result = check_symptoms("Tengo hinchazon y problemas de digestion")

    assert result is not None
    assert result["axis"] == "microbiota intestinal"


def test_check_symptoms_returns_none_when_no_keyword_matches():
    result = check_symptoms("Quiero saber mas sobre nutricion en general")

    assert result is None


def test_check_symptoms_is_case_insensitive():
    result = check_symptoms("FATIGA y ESTRES constantes")

    assert result is not None
