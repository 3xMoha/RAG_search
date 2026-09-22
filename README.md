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

For local answers, install [Ollama](https://ollama.com/download).
Run `ollama serve` in another terminal if it isn't already running, then:

```bash
ollama pull qwen3.5:4b  # one-time download, about 3.4 GB
uv run rag-search "What is RAG?" --answer --top-k 3
```

Omit the question for interactive answers. Switch models with `--model NAME`
(download it first); change servers with `--ollama-url URL`.
Generation uses a 4,096-token context; keep retrieved passages short.
Answers include source labels but can still be wrong. Tests mock Ollama.

Run tests with:

```bash
uv sync --group test
uv run pytest
```

GitHub Actions runs the tests on pushes and pull requests (downloads the embedding model).

Omit the query for interactive search; Esc, Ctrl+C, or Ctrl+D exits.

The CLI saves chunks, vectors, and settings in `data/.rag-index.json` (under `--data`).
Later launches reuse it; the model still loads to embed questions.
After editing documents or changing chunk/model settings, rebuild:

```bash
uv run rag-search --rebuild
```

Evaluate 12 fixed questions against a separate sample corpus:

```bash
uv run python eval/run.py
```

Hit@3 counts questions with the expected source in the top three chunks.
No-answer rejections count unrelated questions returning no matches.
Compare runs with `--max-chars 200` or `--min-score 0.4`.
This small set is a learning baseline, not a general quality benchmark.
