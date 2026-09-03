import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag.chunker import Document, chunk_document, chunk_documents, load_documents


class ChunkerTests(unittest.TestCase):
    def test_load_documents_reads_supported_files_and_keeps_source(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "notes.md").write_text("A note", encoding="utf-8")
            (directory / "ignored.csv").write_text("not a document", encoding="utf-8")

            documents = load_documents(directory)

        self.assertEqual(documents, [Document(source="notes.md", text="A note")])

    def test_chunk_document_groups_paragraphs_and_assigns_ids(self):
        document = Document(source="notes.md", text="First paragraph.\n\nSecond paragraph.")

        chunks = chunk_document(document, max_chars=20)

        self.assertEqual([chunk.chunk_id for chunk in chunks], [0, 1])
        self.assertEqual([chunk.text for chunk in chunks], ["First paragraph.", "Second paragraph."])
        self.assertTrue(all(chunk.source == "notes.md" for chunk in chunks))

    def test_chunk_documents_processes_multiple_documents(self):
        documents = [
            Document(source="one.md", text="One"),
            Document(source="two.md", text="Two"),
        ]

        chunks = chunk_documents(documents)

        self.assertEqual([(chunk.source, chunk.chunk_id) for chunk in chunks], [("one.md", 0), ("two.md", 0)])

    def test_max_chars_must_be_positive(self):
        with self.assertRaises(ValueError):
            chunk_document(Document(source="notes.md", text="Text"), max_chars=0)


if __name__ == "__main__":
    unittest.main()