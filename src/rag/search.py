from pathlib import Path

from rag.chunker import chunk_documents, load_documents
from rag.embedder import Embedder
from rag.retriever import rank_chunks


def search(
    directory: Path,
    query: str,
    embedder: Embedder | None = None,
    max_chars: int = 500,
    top_k: int = 3,
):
    """Find the chunks most relevant to a query in a document directory."""
    documents = load_documents(directory)
    chunks = chunk_documents(documents, max_chars=max_chars)

    embedder = embedder or Embedder()
    chunk_vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    query_vector = embedder.embed_query(query)

    return rank_chunks(chunks, chunk_vectors, query_vector, top_k=top_k)