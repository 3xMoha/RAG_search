import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.embedder import Embedder
from rag.search import search


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder()

    def test_real_documents_can_be_searched(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "notes.md").write_text(
                "Retrieval-augmented generation gives a language model relevant information at question time.",
                encoding="utf-8",
            )
            results = search(
                directory,
                "What is retrieval augmented generation?",
                embedder=self.embedder,
                max_chars=150,
                top_k=1,
            )

        best_chunk, score = results[0]
        self.assertEqual(best_chunk.source, "notes.md")
        self.assertIn("Retrieval-augmented generation", best_chunk.text)
        self.assertGreater(score, 0.4)


if __name__ == "__main__":
    unittest.main()
