"""DeepSeek adapter for the project's common ``lm(messages) -> str`` API.

Besides sending requests, this module keeps a local response cache and a
JSONL usage ledger.  Token counts come from DeepSeek's response; monetary
costs are estimates based on configurable per-million-token prices.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from cs329z_hw1.types import Message


REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_ROLES = {"system", "user", "assistant"}

# Current official peak/off-peak rates in CNY per one million tokens.
# They are only defaults for estimates and can be overridden through .env.
_DEFAULT_PRICES_CNY = {
    "deepseek-flash": {
        "peak": (0.04, 2.0, 8.0),
        "off_peak": (0.02, 1.0, 4.0),
    },
    "deepseek-v4-pro": {
        "peak": (0.30, 9.0, 27.0),
        "off_peak": (0.15, 4.5, 13.5),
    },
}

_ledger_lock = threading.Lock()


class DeepSeekBudgetExceeded(RuntimeError):
    """Raised before an API call when the configured estimated budget is met."""


class DeepSeekOutputTruncated(RuntimeError):
    """Raised when the provider reports that the response hit max_tokens."""

    def __init__(self, text: str, max_tokens: int) -> None:
        self.text = text
        self.max_tokens = max_tokens
        super().__init__(
            f"DeepSeek stopped at max_tokens={max_tokens}; increase max_tokens "
            "or request a shorter response."
        )


def _cache_dir() -> Path:
    return Path(
        os.environ.get(
            "DEEPSEEK_CACHE_DIR",
            REPO_ROOT / ".lm_cache" / "deepseek",
        )
    )


def _ledger_path() -> Path:
    return Path(
        os.environ.get(
            "DEEPSEEK_LEDGER_PATH",
            REPO_ROOT / ".lm_cache" / "deepseek_ledger.jsonl",
        )
    )


def _validate_messages(messages: Sequence[Message]) -> list[Message]:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a non-empty list")

    copied: list[Message] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be a dict")
        if set(message) != {"role", "content"}:
            raise ValueError("each message must contain only role and content")
        if message["role"] not in VALID_ROLES:
            raise ValueError(f"unsupported message role: {message['role']!r}")
        if not isinstance(message["content"], str):
            raise ValueError("message content must be a string")
        copied.append({"role": message["role"], "content": message["content"]})

    return copied


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _canonical_price_model(model: str) -> str | None:
    if model in {
        "deepseek-flash",
        "deepseek-v4-flash",
        "deepseek-v4-flash-vision-exp",
    }:
        return "deepseek-flash"
    if model == "deepseek-v4-pro":
        return "deepseek-v4-pro"
    return None


def _resolve_prices(model: str) -> tuple[float, float, float] | None:
    """Return (cached input, uncached input, output) CNY prices per 1M tokens."""
    names = (
        "DEEPSEEK_PRICE_CACHED_INPUT_CNY_PER_MILLION",
        "DEEPSEEK_PRICE_INPUT_CNY_PER_MILLION",
        "DEEPSEEK_PRICE_OUTPUT_CNY_PER_MILLION",
    )
    configured = [os.environ.get(name) for name in names]
    if any(value is not None for value in configured):
        if not all(value is not None for value in configured):
            raise ValueError(
                "Set all three DEEPSEEK_PRICE_*_CNY_PER_MILLION values, or none."
            )
        return tuple(float(value) for value in configured)  # type: ignore[arg-type,return-value]

    canonical = _canonical_price_model(model)
    if canonical is None:
        return None

    tier = os.environ.get("DEEPSEEK_PRICE_TIER", "peak").strip().lower()
    if tier not in {"peak", "off_peak"}:
        raise ValueError("DEEPSEEK_PRICE_TIER must be 'peak' or 'off_peak'")
    return _DEFAULT_PRICES_CNY[canonical][tier]


def _estimate_cost_cny(
    prices: tuple[float, float, float] | None,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float | None:
    if prices is None:
        return None
    cached_price, input_price, output_price = prices
    cached = min(max(cached_input_tokens, 0), max(input_tokens, 0))
    uncached = max(input_tokens - cached, 0)
    return (
        cached * cached_price
        + uncached * input_price
        + max(output_tokens, 0) * output_price
    ) / 1_000_000


def _read_ledger() -> list[dict[str, Any]]:
    path = _ledger_path()
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def estimated_spend_cny() -> float:
    """Return the sum of known estimated costs in the DeepSeek ledger."""
    return sum(
        float(row["estimated_cost_cny"])
        for row in _read_ledger()
        if isinstance(row.get("estimated_cost_cny"), (int, float))
    )


def _append_ledger(row: dict[str, Any]) -> None:
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ledger_lock:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


class DeepSeekLM:
    """A callable DeepSeek client compatible with ``LMCallable``.

    Thinking mode is disabled by default because the first pipelines are
    simple classification and summarization tasks.  It can be enabled for
    harder agent work with ``thinking=True`` or ``DEEPSEEK_THINKING=true``.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        max_tokens: int = 2048,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        cache: bool = True,
        tag: str = "",
    ) -> None:
        load_dotenv(REPO_ROOT / ".env")

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set in root/.env")

        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        self.base_url = os.environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        )
        self.max_tokens = max_tokens
        self.thinking = (
            _env_bool("DEEPSEEK_THINKING", False)
            if thinking is None
            else thinking
        )
        self.reasoning_effort = reasoning_effort or os.environ.get(
            "DEEPSEEK_REASONING_EFFORT",
            "high",
        )
        self.cache = cache
        self.tag = tag
        self.prices = _resolve_prices(self.model)

        if os.environ.get("DEEPSEEK_BUDGET_CNY") and self.prices is None:
            raise ValueError(
                "DEEPSEEK_BUDGET_CNY requires known pricing; configure all three "
                "DEEPSEEK_PRICE_*_CNY_PER_MILLION values for this model."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=3,
            timeout=120,
        )

        self.calls = 0
        self.cached_calls = 0
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.estimated_cost_cny = 0.0
        self.seconds = 0.0
        self._totals_lock = threading.Lock()

    def __call__(self, messages: list[Message]) -> str:
        clean_messages = _validate_messages(messages)
        key = self._cache_key(clean_messages)
        cache_path = _cache_dir() / key[:2] / f"{key}.json"

        with self._totals_lock:
            self.calls += 1

        if self.cache and cache_path.exists():
            with self._totals_lock:
                self.cached_calls += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))["text"]

        self._check_budget()
        started_at = time.time()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": clean_messages,
            "max_tokens": self.max_tokens,
            "stream": False,
            "extra_body": {
                "thinking": {
                    "type": "enabled" if self.thinking else "disabled",
                }
            },
        }
        if self.thinking:
            request["reasoning_effort"] = self.reasoning_effort

        response = self.client.chat.completions.create(**request)
        elapsed = time.time() - started_at
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = self._extract_usage(response)
        cost = _estimate_cost_cny(
            self.prices,
            input_tokens=usage["input_tokens"],
            cached_input_tokens=usage["cached_input_tokens"],
            output_tokens=usage["output_tokens"],
        )

        with self._totals_lock:
            self.input_tokens += usage["input_tokens"]
            self.cached_input_tokens += usage["cached_input_tokens"]
            self.output_tokens += usage["output_tokens"]
            self.reasoning_tokens += usage["reasoning_tokens"]
            if cost is not None:
                self.estimated_cost_cny += cost
            self.seconds += elapsed

        _append_ledger(
            {
                "ts": round(started_at, 3),
                "model": self.model,
                "tag": self.tag,
                **usage,
                "estimated_cost_cny": cost,
                "price_tier": os.environ.get("DEEPSEEK_PRICE_TIER", "peak"),
                "thinking": self.thinking,
                "seconds": round(elapsed, 3),
                "finish_reason": getattr(choice, "finish_reason", None),
            }
        )

        if getattr(choice, "finish_reason", None) == "length":
            raise DeepSeekOutputTruncated(text, self.max_tokens)

        if self.cache and text.strip():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(
                f".{os.getpid()}.{threading.get_ident()}.tmp"
            )
            temporary.write_text(
                json.dumps({"text": text, "model": self.model}, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(cache_path)

        return text

    def usage(self) -> dict[str, Any]:
        """Return in-process totals; local cache hits consume no API tokens."""
        with self._totals_lock:
            return {
                "model": self.model,
                "calls": self.calls,
                "local_cached_calls": self.cached_calls,
                "input_tokens": self.input_tokens,
                "provider_cached_input_tokens": self.cached_input_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "estimated_cost_cny": self.estimated_cost_cny,
                "pricing_known": self.prices is not None,
                "seconds": self.seconds,
            }

    def _cache_key(self, messages: list[Message]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "base_url": self.base_url,
                "max_tokens": self.max_tokens,
                "thinking": self.thinking,
                "reasoning_effort": self.reasoning_effort,
                "messages": messages,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _check_budget(self) -> None:
        budget = os.environ.get("DEEPSEEK_BUDGET_CNY")
        if budget is None:
            return
        spent = estimated_spend_cny()
        if spent >= float(budget):
            raise DeepSeekBudgetExceeded(
                f"Estimated DeepSeek spend is ¥{spent:.4f}, meeting the "
                f"configured ¥{float(budget):.4f} budget."
            )

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            }

        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        cached = getattr(usage, "prompt_cache_hit_tokens", None)
        if cached is None:
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0

        output_details = getattr(usage, "completion_tokens_details", None)
        reasoning = (
            getattr(output_details, "reasoning_tokens", 0)
            if output_details is not None
            else 0
        )

        return {
            "input_tokens": input_tokens,
            "cached_input_tokens": int(cached or 0),
            "output_tokens": output_tokens,
            "reasoning_tokens": int(reasoning or 0),
        }
