import json
from dataclasses import asdict
from pathlib import Path

from rag.chunker import Chunk, chunk_documents, load_documents
from rag.embedder import DEFAULT_MODEL, Embedder
from rag.retriever import rank_chunks


class SearchIndex:
    """Prepare document embeddings once and search them repeatedly."""

    def __init__(
        self, directory: Path, embedder: Embedder | None = None, max_chars: int = 500,
        *, index_path: Path | None = None, rebuild: bool = False,
    ):
        if not directory.is_dir():
            raise ValueError(f"Not a data directory: {directory}")
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        settings = {
            "version": 2,
            "directory": str(directory.resolve()),
            "model": embedder.model_name if embedder is not None else DEFAULT_MODEL,
            "max_chars": max_chars,
        }
        if index_path is not None and index_path.exists() and not rebuild:
            self._load(index_path, settings)
            self.embedder = embedder or Embedder()
            return
        documents = load_documents(directory)
        self.chunks = chunk_documents(documents, max_chars=max_chars)
        if not self.chunks:
            raise ValueError(f"No nonempty .md, .txt, .pdf, or .epub files in {directory}")
        self.embedder = embedder or Embedder()
        self.chunk_vectors = self.embedder.embed_documents([chunk.text for chunk in self.chunks])
        if index_path is not None:
            payload = {
                "settings": settings,
                "chunks": [asdict(chunk) for chunk in self.chunks],
                "vectors": self.chunk_vectors,
            }
            index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _load(self, path: Path, settings: dict) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["settings"] != settings:
                raise ValueError("settings changed")
            self.chunks = [Chunk(**chunk) for chunk in payload["chunks"]]
            self.chunk_vectors = payload["vectors"]
            if not self.chunks or len(self.chunks) != len(self.chunk_vectors):
                raise ValueError("invalid chunk/vector count")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot load index {path}; run with --rebuild ({error})") from error

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
