import chromadb

from agente_ai import config, llm_client


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DATA_DIR))
    return client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)


def retrieve(query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[dict]:
    """Embeds the query and returns the top-k matching chunks with source metadata."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    query_embedding = llm_client.embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return [
        {
            "text": doc,
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "source_url": meta["source_url"],
        }
        for doc, meta in zip(documents, metadatas)
    ]
