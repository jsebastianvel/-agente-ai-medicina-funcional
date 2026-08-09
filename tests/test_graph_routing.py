from agente_ai import config
from agente_ai.graph.nodes import route_after_critic, route_after_router, route_after_validate
from agente_ai.graph.schemas import CriticVerdict, RouteDecision

FAKE_CHUNK = {"text": "contenido", "doc_id": "d1", "title": "T", "source_url": "https://x"}


# --- pure routing-function unit tests (no mocking needed) ---


def test_route_after_validate_valid_goes_to_router():
    assert route_after_validate({"query_valid": True}) == "router"


def test_route_after_validate_invalid_goes_to_reject():
    assert route_after_validate({"query_valid": False}) == "reject_response"


def test_route_after_router_passes_through_route():
    assert route_after_router({"route": "symptom_check"}) == "symptom_check"


def test_route_after_critic_approved():
    assert route_after_critic({"critic_verdict": "approved", "revision_count": 1}) == "approved"


def test_route_after_critic_revise_when_under_limit():
    assert route_after_critic({"critic_verdict": "revise", "revision_count": 1}) == "revise"


def test_route_after_critic_insufficient_evidence_when_limit_reached():
    state = {"critic_verdict": "revise", "revision_count": config.MAX_REVISIONS}
    assert route_after_critic(state) == "insufficient_evidence"


# --- full graph, wired end-to-end with every LLM call mocked ---


def _fake_generate_structured(route="rag_qa", critic_verdicts=("approved",)):
    verdicts = iter(critic_verdicts)

    def side_effect(prompt, response_schema, system_instruction=None, max_output_tokens=None):
        if response_schema is RouteDecision:
            return RouteDecision(route=route, reasoning="test")
        if response_schema is CriticVerdict:
            return CriticVerdict(verdict=next(verdicts), feedback="test feedback")
        raise AssertionError(f"unexpected schema {response_schema}")

    return side_effect


def test_full_graph_rag_qa_happy_path(mocker):
    mocker.patch("agente_ai.graph.nodes.rag_retrieve", return_value=[FAKE_CHUNK])
    mocker.patch(
        "agente_ai.llm_client.generate_structured", side_effect=_fake_generate_structured()
    )
    mocker.patch("agente_ai.llm_client.generate_text", return_value="Respuesta generada.")

    from agente_ai.graph.build_graph import graph

    result = graph.invoke({"query": "¿Que es la medicina funcional?"})

    assert result["route"] == "rag_qa"
    assert result["tool_calls_used"] == ["retrieval_tool"]
    assert result["final_answer"] == "Respuesta generada."
    assert result["citations"] == ["T"]


def test_full_graph_revision_loop_then_approves(mocker):
    mocker.patch("agente_ai.graph.nodes.rag_retrieve", return_value=[FAKE_CHUNK])
    mocker.patch(
        "agente_ai.llm_client.generate_structured",
        side_effect=_fake_generate_structured(critic_verdicts=("revise", "approved")),
    )
    mocker.patch("agente_ai.llm_client.generate_text", return_value="Respuesta generada.")

    from agente_ai.graph.build_graph import graph

    result = graph.invoke({"query": "¿Que es la medicina funcional?"})

    assert result["revision_count"] == 2
    assert result["critic_verdict"] == "approved"
    assert result["final_answer"] == "Respuesta generada."


def test_full_graph_gives_up_after_max_revisions(mocker):
    mocker.patch("agente_ai.graph.nodes.rag_retrieve", return_value=[FAKE_CHUNK])
    mocker.patch(
        "agente_ai.llm_client.generate_structured",
        side_effect=_fake_generate_structured(critic_verdicts=("revise", "revise")),
    )
    mocker.patch("agente_ai.llm_client.generate_text", return_value="Respuesta generada.")

    from agente_ai.graph.build_graph import graph

    result = graph.invoke({"query": "¿Que es la medicina funcional?"})

    assert result["revision_count"] == config.MAX_REVISIONS
    assert result["final_answer"] != "Respuesta generada."  # falls back to the honest message
    assert result["citations"] == []


def test_full_graph_reject_path_skips_retrieval_and_critic(mocker):
    mocker.patch(
        "agente_ai.llm_client.generate_structured",
        return_value=RouteDecision(route="reject", reasoning="off-topic"),
    )
    retrieve_mock = mocker.patch("agente_ai.graph.nodes.rag_retrieve")
    generate_text_mock = mocker.patch("agente_ai.llm_client.generate_text")

    from agente_ai.graph.build_graph import graph

    result = graph.invoke({"query": "¿Cual es la capital de Francia?"})

    assert result["route"] == "reject"
    assert result["citations"] == []
    retrieve_mock.assert_not_called()
    generate_text_mock.assert_not_called()


def test_full_graph_symptom_check_composes_with_retrieval(mocker):
    mocker.patch("agente_ai.graph.nodes.rag_retrieve", return_value=[FAKE_CHUNK])
    mocker.patch(
        "agente_ai.llm_client.generate_structured",
        side_effect=_fake_generate_structured(route="symptom_check"),
    )
    mocker.patch("agente_ai.llm_client.generate_text", return_value="Respuesta generada.")

    from agente_ai.graph.build_graph import graph

    result = graph.invoke({"query": "Tengo fatiga cronica e insomnio"})

    assert result["route"] == "symptom_check"
    assert result["tool_calls_used"] == ["symptom_checker_tool", "retrieval_tool"]
    assert result["symptom_match"] is not None
