from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from zipfile import BadZipFile

from bs4 import BeautifulSoup
from ebooklib import epub
from lxml.etree import LxmlError
from pypdf import PdfReader
from pypdf.errors import PyPdfError


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
    """Read text files, PDF pages, and EPUB chapters with source references."""
    documents = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            documents.extend(load_pdf(path))
            continue
        if path.suffix.lower() == ".epub":
            documents.extend(load_epub(path))
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(source=path.name, text=text))

    return documents


def load_epub(path: Path) -> list[Document]:
    """Read chapters in spine order, using headings and paths as references."""
    documents = []
    try:
        book = epub.read_epub(str(path))
        for item_id, _ in book.spine:
            item = book.get_item_with_id(item_id)
            if not isinstance(item, epub.EpubHtml) or isinstance(item, epub.EpubNav):
                continue
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            for tag in soup(["script", "style", "nav"]):
                tag.decompose()
            heading = soup.find(["h1", "h2", "h3"])
            title = heading.get_text(" ", strip=True) if heading else item.get_name()
            for tag in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "br"]):
                tag.insert_before("\n\n")
            text = soup.get_text().strip()
            if text:
                source = f"{path.name}, {title} ({item.get_name()})"
                documents.append(Document(source, text))
    except (epub.EpubException, BadZipFile, LxmlError, KeyError) as error:
        raise ValueError(f"Cannot read EPUB {path.name}: {error}") from error
    if not documents:
        raise ValueError(f"No extractable text in EPUB {path.name}; DRM and image-only books are unsupported")
    return documents


def load_pdf(path: Path) -> list[Document]:
    documents = []
    try:
        with path.open("rb") as stream:
            reader = PdfReader(stream)
            if reader.is_encrypted:
                raise ValueError(f"Encrypted PDF is not supported: {path.name}")
            for number, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(Document(f"{path.name}, page {number}", text))
    except PyPdfError as error:
        raise ValueError(f"Cannot read PDF {path.name}: {error}") from error
    if not documents:
        raise ValueError(f"No extractable text in {path.name}; scanned PDFs need OCR")
    return documents


def chunk_document(document: Document, max_chars: int = 500) -> list[Chunk]:
    """Group paragraphs, splitting oversized ones to respect max_chars."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    paragraphs = [paragraph.strip() for paragraph in document.text.split("\n\n")]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    paragraphs = [
        part for paragraph in paragraphs
        for part in (wrap(paragraph, width=max_chars) if len(paragraph) > max_chars else [paragraph])
    ]
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
