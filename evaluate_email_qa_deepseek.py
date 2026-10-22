"""Evaluate the multi-hop email QA pipeline with DeepSeek.

The evaluator runs questions sequentially so that every BM25 query and its
top results remain easy to inspect while tuning the retrieval prompt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cs329z_hw1 import adapters, data
from cs329z_hw1.deepseek import DeepSeekLM
from tests.helpers import contains_any


FIXTURE_PATH = Path("tests/fixtures/email_qa.json")
MIN_ACCURACY = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate email_qa on the provided gold questions using DeepSeek."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="number of questions to evaluate (1-25, default: 5)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="enable DeepSeek thinking mode (disabled by default)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="maximum output tokens for each model call (default: 2048)",
    )
    return parser.parse_args()


def print_search_trace(search_log: list[dict[str, Any]]) -> None:
    """Print the queries and leading BM25 results for one question."""

    for hop, entry in enumerate(search_log, start=1):
        hits = entry["hits"]
        print(f"  Search {hop}: {entry['query']!r} (k={entry['k']}, {len(hits)} results)")
        for hit in hits[:5]:
            subject = hit["email"].get("subject") or "(no subject)"
            print(
                f"    {hit['email']['id']}  score={hit['score']:.3f}  "
                f"subject={subject}"
            )
        if len(hits) > 5:
            print(f"    ... and {len(hits) - 5} more")


def main() -> None:
    args = parse_args()
    questions = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not 1 <= args.limit <= len(questions):
        raise SystemExit(f"--limit must be between 1 and {len(questions)}")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    archive = data.load_emails()
    print(f"Building BM25 index over {len(archive)} emails...", flush=True)
    started = time.perf_counter()
    index = adapters.run_bm25_build(
        [data.email_text(email) for email in archive]
    )
    print(f"Index built in {time.perf_counter() - started:.2f} seconds.\n")

    lm = DeepSeekLM(
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        tag=f"email_qa/eval-{args.limit}",
    )

    correct_count = 0
    correct_with_support_count = 0
    failed: list[str] = []

    for number, item in enumerate(questions[: args.limit], start=1):
        search_log: list[dict[str, Any]] = []

        def search(query: str, k: int = 5) -> list[dict]:
            scored_hits = adapters.run_bm25_search(index, query, k)
            hits = [
                {"email": archive[doc_id], "score": score}
                for doc_id, score in scored_hits
            ]
            search_log.append({"query": query, "k": k, "hits": hits})
            return [hit["email"] for hit in hits]

        started = time.perf_counter()
        result = adapters.run_email_qa(item["question"], search, lm)
        elapsed = time.perf_counter() - started

        answer = result.get("answer")
        support = result.get("support") or []
        answer_correct = isinstance(answer, str) and contains_any(
            answer, item["answers"]
        )
        support_correct = bool(set(support) & set(item["support"]))
        passed = answer_correct and support_correct

        if answer_correct:
            correct_count += 1
            if support_correct:
                correct_with_support_count += 1
        if not passed:
            failed.append(item["id"])

        print("=" * 80)
        print(
            f"[{number}/{args.limit}] {item['id']} "
            f"({item['hops']}-hop): {item['question']}"
        )
        print_search_trace(search_log)
        print(f"  Answer: {answer!r}")
        print(f"  Support: {support}")
        print(f"  Accepted answers: {item['answers']}")
        print(f"  Gold support: {item['support']}")
        print(f"  Answer correct: {answer_correct}")
        print(f"  Support correct: {support_correct}")
        print(f"  Result: {'PASS' if passed else 'FAIL'} ({elapsed:.2f} seconds)")

    accuracy = correct_count / args.limit
    missing_support = correct_count - correct_with_support_count
    passed_threshold = (
        accuracy >= MIN_ACCURACY
        and correct_count > 0
        and missing_support == 0
    )

    print("\n" + "=" * 80)
    print(f"Thinking mode: {'enabled' if args.thinking else 'disabled'}")
    print(
        f"Answer accuracy: {correct_count}/{args.limit} "
        f"({accuracy:.1%}; target at least {MIN_ACCURACY:.0%})"
    )
    print(
        "Correct answers with gold support: "
        f"{correct_with_support_count}/{correct_count}"
    )
    print(f"Failed questions: {failed or 'none'}")
    print("Usage:", json.dumps(lm.usage(), ensure_ascii=False, indent=2))
    print(f"Overall result: {'PASS' if passed_threshold else 'FAIL'}")

    if not passed_threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
