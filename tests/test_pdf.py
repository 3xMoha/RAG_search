from unittest.mock import Mock

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag.chunker import Document, chunk_document, load_documents
from rag.search import SearchIndex


def write_pdf(path, pages, password=None):
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    if password:
        writer.encrypt(password)
    writer.write(path)


def test_pdf_pages_survive_search_and_reload(tmp_path):
    write_pdf(tmp_path / "book.PDF", ["First page", "", "Third page"])
    assert load_documents(tmp_path) == [
        Document("book.PDF, page 1", "First page"),
        Document("book.PDF, page 3", "Third page"),
    ]
    embedder = Mock(model_name="fake")
    embedder.embed_documents.return_value = [[1, 0], [0, 1]]
    embedder.embed_query.return_value = [0, 1]
    path = tmp_path / ".rag-index.json"
    SearchIndex(tmp_path, embedder, index_path=path)
    loaded = SearchIndex(tmp_path, embedder, index_path=path)
    results = loaded.search("Third page", top_k=1)
    assert results[0][0].source == "book.PDF, page 3"
    embedder.embed_documents.assert_called_once()


@pytest.mark.parametrize("kind, message", [
    ("blank", "need OCR"), ("encrypted", "Encrypted PDF"), ("broken", "Cannot read PDF"),
])
def test_unusable_pdf(tmp_path, kind, message):
    path = tmp_path / "book.pdf"
    if kind == "broken":
        path.write_bytes(b"not a PDF")
    else:
        write_pdf(path, [""], password="secret" if kind == "encrypted" else None)
    with pytest.raises(ValueError, match=message):
        load_documents(tmp_path)


def test_long_paragraph_preserves_words_and_source():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_document(Document("book.pdf, page 2", text), max_chars=50)
    assert all(len(chunk.text) <= 50 for chunk in chunks)
    assert " ".join(chunk.text for chunk in chunks).split() == text.split()
    assert all(chunk.source == "book.pdf, page 2" for chunk in chunks)
