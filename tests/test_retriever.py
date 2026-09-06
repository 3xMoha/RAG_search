import unittest

from rag.chunker import Chunk
from rag.retriever import cosine_similarity, rank_chunks


class RetrieverTests(unittest.TestCase):
    def test_cosine_similarity_is_one_for_identical_direction(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [2, 0]), 1.0)

    def test_cosine_similarity_is_zero_for_perpendicular_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_rank_chunks_returns_most_similar_first(self):
        chunks = [
            Chunk(source="notes.md", chunk_id=0, text="About cooking"),
            Chunk(source="notes.md", chunk_id=1, text="About RAG"),
        ]
        chunk_vectors = [[1, 0], [0, 1]]

        results = rank_chunks(chunks, chunk_vectors, query_vector=[0, 1], top_k=1)

        self.assertEqual(results, [(chunks[1], 1.0)])

    def test_rank_chunks_requires_one_vector_per_chunk(self):
        with self.assertRaises(ValueError):
            rank_chunks(
                [Chunk(source="notes.md", chunk_id=0, text="Text")],
                [],
                query_vector=[1, 0],
            )


if __name__ == "__main__":
    unittest.main()