import unittest
from pathlib import Path

from rag.embedder import Embedder
from rag.search import search


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder()

    def test_real_documents_can_be_searched(self):
        results = search(
            Path("data"),
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