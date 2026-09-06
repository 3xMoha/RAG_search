import math
import unittest

from rag.embedder import Embedder


class EmbedderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = Embedder()

    def test_document_embeddings_have_expected_shape_and_length(self):
        vectors = self.embedder.embed_documents(["RAG retrieves useful context.", "Embeddings are vectors."])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 384)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vectors[0])), 1.0, places=5)

    def test_query_embedding_has_same_dimension(self):
        vector = self.embedder.embed_query("What is RAG?")

        self.assertEqual(len(vector), 384)


if __name__ == "__main__":
    unittest.main()