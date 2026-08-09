from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    source_url: str
    body: str


def load_documents(raw_dir: Path) -> list[Document]:
    """Parses the `TITULO: .../FUENTE: .../---/<body>` .txt format used in data/raw.
    Uses utf-8-sig since one source file has a BOM on its first line."""
    documents = []
    for path in sorted(Path(raw_dir).glob("*.txt")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        title = lines[0].removeprefix("TITULO:").strip()
        source_url = lines[1].removeprefix("FUENTE:").strip()
        separator_idx = next(i for i, line in enumerate(lines) if line.strip() == "---")
        body = "\n".join(lines[separator_idx + 1 :]).strip()
        documents.append(Document(doc_id=path.stem, title=title, source_url=source_url, body=body))
    return documents
