from unittest.mock import Mock

import pytest
from ebooklib import epub

from rag.chunker import load_documents
from rag.search import SearchIndex


def write_epub(path, contents):
    book = epub.EpubBook()
    book.set_identifier("test")
    book.set_title("Test book")
    book.set_language("en")
    chapters = []
    for number, content in enumerate(contents):
        chapter = epub.EpubHtml(uid=f"c{number}", file_name=f"c{number}.xhtml")
        chapter.content = content
        book.add_item(chapter)
        chapters.append(chapter)
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *reversed(chapters)]
    epub.write_epub(str(path), book)


def test_epub_reading_order_text_and_saved_references(tmp_path):
    write_epub(tmp_path / "book.EPUB", [
        "<h1>First</h1><p>Some <em>important</em> text &amp; details.</p><script>hidden</script>",
        "<p>Second section.</p>",
    ])
    documents = load_documents(tmp_path)
    assert len(documents) == 2
    assert documents[0].source == "book.EPUB, c1.xhtml (c1.xhtml)"
    assert documents[1].source == "book.EPUB, First (c0.xhtml)"
    assert "Some important text & details." in documents[1].text
    assert "hidden" not in documents[1].text
    embedder = Mock(model_name="fake")
    embedder.embed_documents.return_value = [[1, 0], [0, 1]]
    embedder.embed_query.return_value = [0, 1]
    path = tmp_path / ".rag-index.json"
    SearchIndex(tmp_path, embedder, index_path=path)
    index = SearchIndex(tmp_path, embedder, index_path=path)
    assert index.search("important", top_k=1)[0][0].source == documents[1].source


def test_empty_epub(tmp_path):
    write_epub(tmp_path / "empty.epub", ["<p> </p>"])
    with pytest.raises(ValueError, match="No extractable text"):
        load_documents(tmp_path)


def test_invalid_epub(tmp_path):
    (tmp_path / "broken.epub").write_bytes(b"not an EPUB")
    with pytest.raises(ValueError, match="Cannot read EPUB broken.epub"):
        load_documents(tmp_path)
