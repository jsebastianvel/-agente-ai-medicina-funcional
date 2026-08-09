import logging

import chromadb

from agente_ai import config, llm_client
from agente_ai.ingestion.chunker import chunk_documents
from agente_ai.ingestion.loader import load_documents

logger = logging.getLogger(__name__)


def build_index() -> int:
    """Loads the raw corpus, chunks it, embeds each chunk with Gemini, and
    upserts it into the local Chroma collection. Returns the chunk count."""
    docs = load_documents(config.RAW_DATA_DIR)
    if not docs:
        raise RuntimeError(f"No documents found in {config.RAW_DATA_DIR}")
    chunks = chunk_documents(docs)

    logger.info("Embedding %d chunks from %d documents...", len(chunks), len(docs))
    embeddings = [llm_client.embed_text(c.text) for c in chunks]

    config.CHROMA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DATA_DIR))
    collection = client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"doc_id": c.doc_id, "title": c.title, "source_url": c.source_url} for c in chunks
        ],
    )
    logger.info("Indexed %d chunks into collection '%s'.", len(chunks), config.CHROMA_COLLECTION_NAME)
    return len(chunks)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    build_index()
