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

    def test_rank_chunks_excludes_scores_below_threshold(self):
        chunks = [
            Chunk(source="notes.md", chunk_id=0, text="Relevant"),
            Chunk(source="notes.md", chunk_id=1, text="Irrelevant"),
        ]

        results = rank_chunks(
            chunks,
            [[1, 0], [0, 1]],
            query_vector=[1, 0],
            min_score=0.5,
        )

        self.assertEqual(results, [(chunks[0], 1.0)])

    def test_min_score_must_be_a_valid_cosine_score(self):
        with self.assertRaises(ValueError):
            rank_chunks([], [], query_vector=[1, 0], min_score=1.1)


if __name__ == "__main__":
    unittest.main()