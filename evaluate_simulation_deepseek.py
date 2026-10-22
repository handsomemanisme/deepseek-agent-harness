"""Run the simulated-user evaluation entirely through DeepSeek.

The course-provided simulation CLI constructs its agent, simulated user, and
judge with the OpenRouter ``LM`` class.  This entry point keeps the original
simulation package unchanged while explicitly injecting ``DeepSeekLM`` for
all four model seats: agent, auxiliary model, simulated user, and judge.

The safe default is a three-persona sample with concurrency one::

    uv run python evaluate_simulation_deepseek.py --slice 3

Run all twenty public personas only after the sample is stable::

    uv run python evaluate_simulation_deepseek.py --all --concurrency 1

Results are written under ``runs/deepseek-sim-<timestamp>/``.  Repeated calls
reuse the DeepSeek response cache unless ``--no-cache`` is supplied.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any, Sequence

from cs329z_hw1 import adapters
from cs329z_hw1.deepseek import (
    DeepSeekBudgetExceeded,
    DeepSeekLM,
    DeepSeekOutputTruncated,
)
from cs329z_hw1.llm import BudgetExceeded
from cs329z_hw1.simulation.evaluate import EVAL_SIZE, points, required_successes
from cs329z_hw1.simulation.persona import Persona, load_personas
from cs329z_hw1.simulation.record import ConversationRecord
from cs329z_hw1.simulation.runner import run_persona


DEFAULT_PERSONA_DIR = Path("tests/fixtures/personas")
DEFAULT_SAMPLE_SIZE = 3


def combine_usage(usages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine DeepSeek usage dictionaries without losing CNY cost data."""

    models = [str(usage.get("model")) for usage in usages if usage.get("model")]
    result: dict[str, Any] = {
        "model": models[0] if models and len(set(models)) == 1 else ", ".join(dict.fromkeys(models)),
        "calls": 0,
        "local_cached_calls": 0,
        "input_tokens": 0,
        "provider_cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "estimated_cost_cny": 0.0,
        "pricing_known": True,
        "seconds": 0.0,
    }
    numeric_keys = (
        "calls",
        "local_cached_calls",
        "input_tokens",
        "provider_cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "estimated_cost_cny",
        "seconds",
    )
    for usage in usages:
        for key in numeric_keys:
            value = usage.get(key, 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] += value
        result["pricing_known"] = bool(result["pricing_known"] and usage.get("pricing_known", False))
    return result


class SimulationDeepSeekLM:
    """DeepSeek callable with simulation-compatible retries and budget stops."""

    def __init__(
        self,
        *,
        model: str | None,
        max_tokens: int,
        retry_max_tokens: int | None = None,
        thinking: bool = False,
        cache: bool = True,
        tag: str,
    ) -> None:
        self.primary = DeepSeekLM(
            model=model,
            max_tokens=max_tokens,
            thinking=thinking,
            cache=cache,
            tag=tag,
        )
        self.retry = (
            DeepSeekLM(
                model=model,
                max_tokens=retry_max_tokens,
                thinking=thinking,
                cache=cache,
                tag=f"{tag}/retry",
            )
            if retry_max_tokens is not None
            else None
        )
        self.model = self.primary.model

    @staticmethod
    def _call(lm: DeepSeekLM, messages: list[dict]) -> str:
        try:
            return lm(messages)
        except DeepSeekBudgetExceeded as exc:
            # The course runner deliberately propagates its BudgetExceeded
            # type, so translate the DeepSeek adapter's equivalent here.
            raise BudgetExceeded(str(exc)) from exc

    def __call__(self, messages: list[dict]) -> str:
        try:
            return self._call(self.primary, messages)
        except DeepSeekOutputTruncated:
            if self.retry is None:
                raise
            try:
                return self._call(self.retry, messages)
            except DeepSeekOutputTruncated as exc:
                # This mirrors the course simulation behavior: after one
                # larger retry, retain the visible portion instead of failing.
                return exc.text

    def usage(self) -> dict[str, Any]:
        lms = [self.primary]
        if self.retry is not None:
            lms.append(self.retry)
        return combine_usage([lm.usage() for lm in lms])


def estimated_cost_cny(record: ConversationRecord) -> float:
    return sum(
        float(usage.get("estimated_cost_cny", 0.0))
        for usage in record.usage.values()
        if isinstance(usage, dict)
    )


def tool_count(record: ConversationRecord) -> int:
    return sum(
        1
        for session in record.sessions
        for event in session.transcript
        if event.get("type") == "tool_call"
    )


def format_cny_table(records: Sequence[ConversationRecord], *, verbose: bool = False) -> str:
    """Format the normal simulation report using DeepSeek's CNY ledger."""

    width = max([len(record.persona_id) for record in records] + [7])
    lines = [
        f"{'persona':<{width}}  {'category':<15}{'result':<7}"
        f"{'turns':>5}{'tools':>6}{'new CNY':>10}{'sec':>7}"
    ]
    for record in records:
        lines.append(
            f"{record.persona_id:<{width}}  {record.category:<15}"
            f"{'PASS' if record.success else 'FAIL':<7}"
            f"{record.user_turns:>5}{tool_count(record):>6}"
            f"{estimated_cost_cny(record):>10.4f}{record.seconds:>7.1f}"
        )
        if record.error:
            lines.append("    agent raised: " + record.error.strip().splitlines()[-1])
        for check in record.failed_checks:
            lines.append(f"    check failed: {check.type}: {check.detail}")
        if record.judge is not None and not record.error and (verbose or not record.judge.passed):
            lines.append(
                f"    judge {'PASS' if record.judge.passed else 'FAIL'}: "
                f"{record.judge.reason}"
            )

    successes = sum(record.success for record in records)
    needed = required_successes(len(records))
    total_cost = sum(estimated_cost_cny(record) for record in records)
    lines.append("")
    lines.append(
        f"Successful conversations: {successes} of {len(records)} "
        f"(full credit at this size needs {needed}). "
        f"Estimated new spending: ¥{total_cost:.4f}."
    )
    if len(records) == EVAL_SIZE:
        lines.append(f"Points under the handout's rule: {points(successes)} of 10.")
    lines.append(
        "A cached response increases local_cached_calls but adds no new token cost. "
        "CNY amounts are estimates; the DeepSeek console is authoritative."
    )
    return "\n".join(lines)


def make_models(persona: Persona, args: argparse.Namespace) -> tuple[
    SimulationDeepSeekLM,
    SimulationDeepSeekLM,
    SimulationDeepSeekLM,
    SimulationDeepSeekLM,
]:
    prefix = f"deepseek-sim/{persona.id}"
    common = {"model": args.model, "cache": not args.no_cache}
    agent = SimulationDeepSeekLM(
        **common,
        max_tokens=args.agent_max_tokens,
        thinking=args.thinking,
        tag=f"{prefix}/agent",
    )
    aux = SimulationDeepSeekLM(
        **common,
        max_tokens=args.aux_max_tokens,
        thinking=False,
        tag=f"{prefix}/aux",
    )
    user = SimulationDeepSeekLM(
        **common,
        max_tokens=args.user_max_tokens,
        retry_max_tokens=2 * args.user_max_tokens,
        thinking=False,
        tag=f"{prefix}/user",
    )
    judge = SimulationDeepSeekLM(
        **common,
        max_tokens=args.judge_max_tokens,
        retry_max_tokens=2 * args.judge_max_tokens,
        thinking=False,
        tag=f"{prefix}/judge",
    )
    return agent, aux, user, judge


def evaluate_one(
    persona: Persona,
    args: argparse.Namespace,
    out_dir: Path,
) -> ConversationRecord:
    agent, aux, user, judge = make_models(persona, args)

    def make_session(config):
        configured = dataclasses.replace(config, aux_lm=aux)
        return adapters.run_agent_session(agent, None, configured)

    record = run_persona(
        persona,
        make_session,
        user_lm=user,
        judge_lm=judge,
        agent_lms=(),
        memory_out=out_dir / f"{persona.id}.memory",
    )
    record.usage = {
        "agent": combine_usage([agent.usage(), aux.usage()]),
        "user": user.usage(),
        "judge": judge.usage(),
    }
    (out_dir / f"{persona.id}.json").write_text(
        record.to_json(),
        encoding="utf-8",
    )
    return record


def run_evaluation(
    personas: Sequence[Persona],
    args: argparse.Namespace,
    out_dir: Path,
) -> list[ConversationRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(persona: Persona) -> ConversationRecord:
        return evaluate_one(persona, args, out_dir)

    ordered: dict[int, ConversationRecord] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(one, persona): (index, persona)
            for index, persona in enumerate(personas)
        }
        for finished, future in enumerate(as_completed(futures), start=1):
            index, persona = futures[future]
            record = future.result()
            ordered[index] = record
            print(
                f"[{finished:02d}/{len(personas):02d}] "
                f"{'PASS' if record.success else 'FAIL'} "
                f"{persona.id} "
                f"(turns={record.user_turns}, tools={tool_count(record)}, "
                f"new=¥{estimated_cost_cny(record):.4f}, {record.seconds:.1f}s)",
                flush=True,
            )
    return [ordered[index] for index in range(len(personas))]


def aggregate_seat_usage(records: Sequence[ConversationRecord], seat: str) -> dict[str, Any]:
    return combine_usage(
        [
            record.usage[seat]
            for record in records
            if isinstance(record.usage.get(seat), dict)
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "personas",
        nargs="?",
        type=Path,
        default=DEFAULT_PERSONA_DIR,
        help="persona JSON directory (default: tests/fixtures/personas)",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--slice",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"run the first N matching personas (default: {DEFAULT_SAMPLE_SIZE})",
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="run every matching persona",
    )
    parser.add_argument(
        "--only",
        help="only personas whose id or category contains this text",
    )
    parser.add_argument("--model", default=None, help="DeepSeek model override")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--agent-max-tokens", type=int, default=4096)
    parser.add_argument("--aux-max-tokens", type=int, default=2048)
    parser.add_argument("--user-max-tokens", type=int, default=1000)
    parser.add_argument("--judge-max-tokens", type=int, default=2500)
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="enable thinking only for the agent model",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="force new API calls instead of using the local DeepSeek cache",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--list",
        action="store_true",
        help="list matching personas without making model calls",
    )
    args = parser.parse_args(argv)

    if not args.all and args.slice <= 0:
        parser.error("--slice must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    limits = (
        args.agent_max_tokens,
        args.aux_max_tokens,
        args.user_max_tokens,
        args.judge_max_tokens,
    )
    if any(limit <= 0 for limit in limits):
        parser.error("all token limits must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    personas = load_personas(args.personas)
    if args.only:
        needle = args.only.casefold()
        personas = [
            persona
            for persona in personas
            if needle in persona.id.casefold() or needle in persona.category.casefold()
        ]
    if not args.all:
        personas = personas[: args.slice]
    if not personas:
        print("No matching persona files found.")
        return 2

    if args.list:
        for persona in personas:
            print(f"{persona.id}\t{persona.category}")
        print(f"{len(personas)} persona(s); no model calls made.")
        return 0

    out_dir = args.out or Path("runs") / time.strftime("deepseek-sim-%Y%m%d-%H%M%S")
    print(
        f"Running {len(personas)} persona(s) with DeepSeek "
        f"(concurrency={args.concurrency}, cache={'off' if args.no_cache else 'on'}).",
        flush=True,
    )
    try:
        records = run_evaluation(personas, args, out_dir)
    except BudgetExceeded as exc:
        print(f"Stopped by DeepSeek budget limit: {exc}")
        print(f"Partial records, if any, are in {out_dir}/")
        return 2

    table = format_cny_table(records, verbose=args.verbose)
    (out_dir / "summary.txt").write_text(table + "\n", encoding="utf-8")

    successes = sum(record.success for record in records)
    report = {
        "configuration": {
            "persona_dir": str(args.personas),
            "persona_ids": [persona.id for persona in personas],
            "model": args.model,
            "concurrency": args.concurrency,
            "thinking": args.thinking,
            "cache": not args.no_cache,
            "agent_max_tokens": args.agent_max_tokens,
            "aux_max_tokens": args.aux_max_tokens,
            "user_max_tokens": args.user_max_tokens,
            "judge_max_tokens": args.judge_max_tokens,
        },
        "successes": successes,
        "total": len(records),
        "required_successes": required_successes(len(records)),
        "points": points(successes, len(records)),
        "estimated_cost_cny": sum(estimated_cost_cny(record) for record in records),
        "usage": {
            seat: aggregate_seat_usage(records, seat)
            for seat in ("agent", "user", "judge")
        },
        "results": [
            {
                "persona_id": record.persona_id,
                "category": record.category,
                "success": record.success,
                "seconds": record.seconds,
                "estimated_cost_cny": estimated_cost_cny(record),
            }
            for record in records
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(table)
    print(f"Records saved in {out_dir}/")
    return 0 if successes >= required_successes(len(records)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
