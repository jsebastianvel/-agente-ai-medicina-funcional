from pathlib import Path

from agente_ai.ingestion.chunker import chunk_document
from agente_ai.ingestion.loader import Document, load_documents

SAMPLE_DOC = """TITULO: Titulo de prueba
FUENTE: https://example.com/articulo
---

Primer parrafo con suficientes palabras como para superar el umbral minimo de
palabras que exige el chunker antes de considerarlo un fragmento propio en
vez de fusionarlo con el anterior.

Segundo parrafo, tambien largo, para que quede como un chunk independiente
en vez de fusionarse con el primero durante la division por parrafos.

Corto.
"""


def test_load_documents_parses_titulo_fuente_body(tmp_path: Path):
    (tmp_path / "sample.txt").write_text(SAMPLE_DOC, encoding="utf-8")

    docs = load_documents(tmp_path)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "sample"
    assert doc.title == "Titulo de prueba"
    assert doc.source_url == "https://example.com/articulo"
    assert doc.body.startswith("Primer parrafo")
    assert "Corto." in doc.body


def test_load_documents_handles_bom(tmp_path: Path):
    (tmp_path / "bom.txt").write_bytes(("﻿" + SAMPLE_DOC).encode("utf-8"))

    docs = load_documents(tmp_path)

    assert docs[0].title == "Titulo de prueba"


def test_chunk_document_splits_on_paragraphs_and_merges_short_fragments():
    doc = Document(
        doc_id="d1",
        title="T",
        source_url="https://x",
        body=(
            "Primer parrafo con suficientes palabras como para superar el umbral "
            "minimo de palabras que exige el chunker antes de considerarlo un "
            "fragmento propio en vez de fusionarlo con el anterior.\n\n"
            "Segundo parrafo, tambien largo, para que quede como un chunk "
            "independiente en vez de fusionarse con el primero durante la "
            "division por parrafos que hace la funcion bajo prueba aqui mismo.\n\n"
            "Corto."
        ),
    )

    chunks = chunk_document(doc, min_words=15)

    # The short trailing paragraph ("Corto.") must merge into the previous chunk,
    # not become its own chunk.
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "d1_0"
    assert chunks[1].text.endswith("Corto.")
    assert all(c.doc_id == "d1" and c.title == "T" for c in chunks)


def test_chunk_document_falls_back_to_whole_body_when_no_paragraphs():
    doc = Document(doc_id="d2", title="T", source_url="https://x", body="Una sola linea sin parrafos.")

    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "Una sola linea sin parrafos."
