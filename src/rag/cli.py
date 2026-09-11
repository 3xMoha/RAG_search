import argparse
from pathlib import Path

from rag.search import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local documents with RAG retrieval.")
    parser.add_argument("query")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    for chunk, score in search(args.data, args.query, top_k=args.top_k):
        print(f"[{score:.3f}] {chunk.source} (chunk {chunk.chunk_id})")
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()