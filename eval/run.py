"""Run from the repository root: uv run python eval/run.py."""

import argparse
import json
from pathlib import Path

from rag.search import SearchIndex


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on a small fixed corpus.")
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--min-score", type=float, default=0.25)
    args = parser.parse_args()
    directory = Path(__file__).parent
    cases = json.loads((directory / "questions.json").read_text())
    index = SearchIndex(directory / "docs", max_chars=args.max_chars)
    hits = rejected = answerable = unanswerable = 0
    for case in cases:
        results = index.search(case["question"], top_k=3, min_score=args.min_score)
        sources = {chunk.source for chunk, _ in results}
        if case["source"] is None:
            unanswerable += 1
            passed = not results
            rejected += passed
        else:
            answerable += 1
            passed = case["source"] in sources
            hits += passed
        print(f"{'PASS' if passed else 'MISS'} {case['question']}")
    print(f"Hit@3: {hits}/{answerable}; no-answer rejections: {rejected}/{unanswerable}")


if __name__ == "__main__":
    main()
