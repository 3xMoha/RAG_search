from pathlib import Path

from rag.chunker import chunk_documents, load_documents
from rag.embedder import Embedder
from rag.retriever import rank_chunks


class SearchIndex:
    """Prepare document embeddings once and search them repeatedly."""

    def __init__(self, directory: Path, embedder: Embedder | None = None, max_chars: int = 500):
        if not directory.is_dir():
            raise ValueError(f"Not a data directory: {directory}")
        documents = load_documents(directory)
        self.chunks = chunk_documents(documents, max_chars=max_chars)
        if not self.chunks:
            raise ValueError(f"No nonempty .md or .txt files in {directory}")
        self.embedder = embedder or Embedder()
        self.chunk_vectors = self.embedder.embed_documents([chunk.text for chunk in self.chunks])

    def search(self, query: str, top_k: int = 3, min_score: float | None = None):
        query_vector = self.embedder.embed_query(query)
        return rank_chunks(
            self.chunks,
            self.chunk_vectors,
            query_vector,
            top_k=top_k,
            min_score=min_score,
        )


def search(
    directory: Path,
    query: str,
    embedder: Embedder | None = None,
    max_chars: int = 500,
    top_k: int = 3,
    min_score: float | None = None,
):
    """Find the chunks most relevant to a query in a document directory."""
    return SearchIndex(directory, embedder=embedder, max_chars=max_chars).search(
        query,
        top_k=top_k,
        min_score=min_score,
    )