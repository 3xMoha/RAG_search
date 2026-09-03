# RAG notes

Retrieval-augmented generation gives a language model relevant information at question time.

The first stage is preparing documents. We load the source text and split it into smaller chunks.

Each chunk keeps its source filename so a later answer can cite where the information came from.