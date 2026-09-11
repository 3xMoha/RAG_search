import math

from rag.chunker import Chunk


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Measure how closely two vectors point in the same direction."""
    if len(first) != len(second):
        raise ValueError("vectors must have the same dimension")

    first_length = math.sqrt(sum(value * value for value in first))
    second_length = math.sqrt(sum(value * value for value in second))
    if first_length == 0 or second_length == 0:
        raise ValueError("vectors must not be zero-length")

    dot_product = sum(first_value * second_value for first_value, second_value in zip(first, second))
    return dot_product / (first_length * second_length)


def rank_chunks(
    chunks: list[Chunk],
    chunk_vectors: list[list[float]],
    query_vector: list[float],
    top_k: int = 3,
    min_score: float | None = None,
) -> list[tuple[Chunk, float]]:
    """Return the chunks whose vectors are most similar to the query."""
    if len(chunks) != len(chunk_vectors):
        raise ValueError("every chunk must have one vector")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if min_score is not None and not -1 <= min_score <= 1:
        raise ValueError("min_score must be between -1 and 1")

    scored_chunks = [
        (chunk, cosine_similarity(vector, query_vector))
        for chunk, vector in zip(chunks, chunk_vectors)
    ]
    if min_score is not None:
        scored_chunks = [result for result in scored_chunks if result[1] >= min_score]
    scored_chunks.sort(key=lambda result: result[1], reverse=True)
    return scored_chunks[:top_k]