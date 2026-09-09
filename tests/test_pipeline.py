import unittest
from pathlib import Path

from rag.chunker import chunk_documents, load_documents
from rag.embedder import Embedder
from rag.retriever import rank_chunks


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder()

    def test_real_documents_can_be_searched(self):
        documents = load_documents(Path("data"))
        chunks = chunk_documents(documents, max_chars=150)
        chunk_vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        query_vector = self.embedder.embed_query("What is retrieval augmented generation?")

        results = rank_chunks(chunks, chunk_vectors, query_vector, top_k=1)

        best_chunk, score = results[0]
        self.assertEqual(best_chunk.source, "notes.md")
        self.assertIn("Retrieval-augmented generation", best_chunk.text)
        self.assertGreater(score, 0.4)


if __name__ == "__main__":
    unittest.main()