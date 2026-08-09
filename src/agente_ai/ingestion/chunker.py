import re
from dataclasses import dataclass

from agente_ai.ingestion.loader import Document

MIN_CHUNK_WORDS = 40
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    source_url: str


def chunk_document(doc: Document, min_words: int = MIN_CHUNK_WORDS) -> list[Chunk]:
    """Splits a document body into paragraph-level chunks, merging any fragment
    shorter than min_words into the previous chunk so retrieval hits stay
    meaningful without falling back to indexing whole articles as one document."""
    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT.split(doc.body) if p.strip()]
    if not paragraphs:
        paragraphs = [doc.body.strip()]

    merged: list[str] = []
    for para in paragraphs:
        if merged and len(para.split()) < min_words:
            merged[-1] = f"{merged[-1]} {para}"
        else:
            merged.append(para)

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}_{i}",
            doc_id=doc.doc_id,
            text=text,
            title=doc.title,
            source_url=doc.source_url,
        )
        for i, text in enumerate(merged)
    ]


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc)]
