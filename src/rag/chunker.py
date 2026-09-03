from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_id: int
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Read Markdown and text files from a directory."""
    documents = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(source=path.name, text=text))

    return documents


def chunk_document(document: Document, max_chars: int = 500) -> list[Chunk]:
    """Group paragraphs into chunks without splitting a paragraph."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    paragraphs = [paragraph.strip() for paragraph in document.text.split("\n\n")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    chunks = []
    current_paragraphs = []
    current_length = 0

    for paragraph in paragraphs:
        separator_length = 2 if current_paragraphs else 0
        would_exceed_limit = current_length + separator_length + len(paragraph) > max_chars

        if current_paragraphs and would_exceed_limit:
            chunks.append(
                Chunk(
                    source=document.source,
                    chunk_id=len(chunks),
                    text="\n\n".join(current_paragraphs),
                )
            )
            current_paragraphs = []
            current_length = 0

        current_paragraphs.append(paragraph)
        current_length += (2 if len(current_paragraphs) > 1 else 0) + len(paragraph)

    if current_paragraphs:
        chunks.append(
            Chunk(
                source=document.source,
                chunk_id=len(chunks),
                text="\n\n".join(current_paragraphs),
            )
        )

    return chunks


def chunk_documents(documents: list[Document], max_chars: int = 500) -> list[Chunk]:
    """Chunk every document while keeping each document's IDs independent."""
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars))
    return chunks