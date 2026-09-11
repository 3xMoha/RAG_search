# RAG Search

A small retrieval-augmented generation project built step by step for learning.

## Use case

Search local Markdown and text files by meaning. The current pipeline loads documents, splits them into chunks, creates embeddings with `all-MiniLM-L6-v2`, and returns the most relevant chunks.

## Run it

```bash
uv sync
uv run rag-search "What is RAG?"
```

Add `.md` or `.txt` files to `data/` and search them with:

```bash
uv run rag-search "your question" --top-k 3
```

Run tests with:

```bash
uv sync --group test
uv run pytest
```