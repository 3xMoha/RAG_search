import argparse
import sys
import termios
import tty
from functools import partial
from pathlib import Path

from rag.search import SearchIndex
from rag.generator import DEFAULT_MODEL, DEFAULT_URL, generate, interpret, warm_up


def read_query() -> str | None:
    """Read one query, returning None when Escape is pressed."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip() if line else None
    print("> ", end="", flush=True)
    characters = []
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            character = sys.stdin.read(1)
            if character in {"", "\x1b", "\x03", "\x04"}:
                print()
                return None
            if character in {"\r", "\n"}:
                print()
                return "".join(characters).strip()
            if character in {"\x7f", "\b"}:
                if characters:
                    characters.pop()
                    print("\b \b", end="", flush=True)
                continue
            characters.append(character)
            print(character, end="", flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def print_results(results) -> None:
    if not results:
        print("No matching results.")
        return
    for chunk, score in results:
        print(f"[{score:.3f}] {chunk.source} (chunk {chunk.chunk_id})")
        print(chunk.text)
        print()


def respond(query, index, top_k, min_score, answer=None, rewrite=None) -> None:
    search_query = rewrite(query) if rewrite else query
    if search_query != query:
        print(f"Searching: {search_query}", flush=True)
    results = index.search(search_query, top_k=top_k, min_score=min_score)
    if answer:
        print(answer(query, results))
    else:
        print_results(results)


def interactive(index: SearchIndex, top_k: int, min_score: float, answer=None, rewrite=None) -> None:
    print("Interactive search. Press Esc to exit.")
    while True:
        query = read_query()
        if query is None:
            return
        if query:
            respond(query, index, top_k, min_score, answer, rewrite)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local documents with RAG retrieval.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max_chars", type=int, default=500)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the saved document index")
    parser.add_argument("--answer", action="store_true", help="Generate an answer with Ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_URL)
    args = parser.parse_args()

    if args.top_k <= 0 or args.max_chars <= 0:
        parser.error("--top-k and --max_chars must be positive")
    if not -1 <= args.min_score <= 1:
        parser.error("--min-score must be between -1 and 1")
    try:
        answer = None
        rewrite = None
        if args.answer:
            answer = partial(generate, model=args.model, url=args.ollama_url)
            rewrite = partial(interpret, model=args.model, url=args.ollama_url)
        index = SearchIndex(
            args.data, max_chars=args.max_chars,
            index_path=args.data / ".rag-index.json", rebuild=args.rebuild,
        )
        if args.answer:
            print("Warming up the LLM...", flush=True)
            warm_up(model=args.model, url=args.ollama_url)
        if args.query:
            respond(args.query, index, args.top_k, args.min_score, answer, rewrite)
        else:
            interactive(index, args.top_k, args.min_score, answer, rewrite)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
