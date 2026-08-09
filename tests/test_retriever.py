from agente_ai.rag import retriever


class FakeCollection:
    def __init__(self, n: int):
        self._n = n

    def count(self):
        return self._n

    def query(self, query_embeddings, n_results):
        docs = ["texto uno", "texto dos"]
        metas = [
            {"doc_id": "doc1", "title": "Titulo 1", "source_url": "https://x/1"},
            {"doc_id": "doc2", "title": "Titulo 2", "source_url": "https://x/2"},
        ]
        return {"documents": [docs[:n_results]], "metadatas": [metas[:n_results]]}


def test_retrieve_returns_empty_list_when_collection_is_empty(mocker):
    mocker.patch("agente_ai.rag.retriever.get_collection", return_value=FakeCollection(n=0))
    mock_embed = mocker.patch("agente_ai.rag.retriever.llm_client.embed_text")

    result = retriever.retrieve("cualquier consulta")

    assert result == []
    mock_embed.assert_not_called()


def test_retrieve_shapes_chunks_with_metadata(mocker):
    mocker.patch("agente_ai.rag.retriever.get_collection", return_value=FakeCollection(n=2))
    mocker.patch("agente_ai.rag.retriever.llm_client.embed_text", return_value=[0.1, 0.2])

    result = retriever.retrieve("consulta", top_k=2)

    assert result == [
        {"text": "texto uno", "doc_id": "doc1", "title": "Titulo 1", "source_url": "https://x/1"},
        {"text": "texto dos", "doc_id": "doc2", "title": "Titulo 2", "source_url": "https://x/2"},
    ]
