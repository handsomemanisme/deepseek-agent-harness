"""Evaluate the daily digest pipeline with DeepSeek on the provided gold days."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs329z_hw1 import adapters, data
from cs329z_hw1.deepseek import DeepSeekLM
from tests.helpers import contains, contains_any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="number of gold days to evaluate (1-5, default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    days = json.loads(
        Path("tests/fixtures/digest_days.json").read_text(encoding="utf-8")
    )
    if not 1 <= args.limit <= len(days):
        raise SystemExit(f"--limit must be between 1 and {len(days)}")

    by_id = {email["id"]: email for email in data.load_emails()}
    lm = DeepSeekLM(
        max_tokens=768,
        thinking=False,
        tag=f"digest/eval-{args.limit}",
    )

    failed_days = 0
    for number, day in enumerate(days[: args.limit], start=1):
        emails = [by_id[email_id] for email_id in day["email_ids"]]
        digest = adapters.run_daily_digest(emails, lm)
        word_count = len(digest.split())

        missing: list[str] = []
        for urgent in day["urgent"]:
            for aliases in urgent["must_mention"]:
                if not contains_any(digest, aliases):
                    missing.append(f"{urgent['id']}: one of {aliases}")

        forbidden: list[str] = []
        for ignored in day["ignore"]:
            for term in ignored["must_not_mention"]:
                if contains(digest, term):
                    forbidden.append(f"{ignored['id']}: {term}")

        length_ok = 0 < word_count <= 300
        passed = length_ok and not missing and not forbidden
        if not passed:
            failed_days += 1

        print("=" * 72)
        print(f"[{number}/{args.limit}] Date: {day['date']}")
        print(f"Words: {word_count} (pass={length_ok})")
        print(f"Missing urgent terms: {missing or 'none'}")
        print(f"Forbidden ignore terms: {forbidden or 'none'}")
        print(f"Day result: {'PASS' if passed else 'FAIL'}")
        print("\nDigest:\n")
        print(digest)

    print("\n" + "=" * 72)
    print(f"Passed days: {args.limit - failed_days}/{args.limit}")
    print("Usage:", lm.usage())

    if failed_days:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
