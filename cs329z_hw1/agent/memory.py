"""Small, file-backed persistent memory for Cardinal agent sessions.

Memory is intentionally deterministic: facts are stored verbatim in a JSONL
file below ``AgentConfig.memory_dir`` and recalled with the project's existing
BM25 implementation.  No language-model call is needed to write or search the
store, so a new session can share memory simply by using the same directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
from typing import Optional

from cs329z_hw1.bm25 import build_bm25, search_bm25


MEMORY_FILENAME = "facts.jsonl"
MAX_RECALL_RESULTS = 5

# A process-local lock is enough for the course harness, whose sessions run in
# one process.  Keeping reads and appends together also prevents two sessions
# from choosing the same monotonically increasing record id.
_memory_lock = threading.RLock()


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One durable fact loaded from the append-only memory file."""

    id: int
    fact: str


def _fallback_terms(text: str) -> set[str]:
    """Return simple Unicode-aware terms for queries BM25 cannot tokenize.

    The course tokenizer deliberately recognizes only ASCII letters and
    digits.  Character terms keep recall useful for short Chinese queries
    without changing the shared BM25 implementation or its tests.
    """

    lowered = text.casefold()
    ascii_terms = set(re.findall(r"[a-z0-9]+", lowered))
    cjk_characters = set(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", lowered))
    return ascii_terms | cjk_characters


class MemoryStore:
    """Append and search facts stored strictly below one memory directory."""

    def __init__(self, memory_dir: Optional[Path]) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir is not None else None

    def _store_path(self, *, create_directory: bool) -> Path:
        if self.memory_dir is None:
            raise RuntimeError("Persistent memory is disabled for this session.")
        if create_directory:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
        elif not self.memory_dir.exists():
            return self.memory_dir / MEMORY_FILENAME

        path = self.memory_dir / MEMORY_FILENAME
        if path.is_symlink():
            raise RuntimeError("The memory store must not be a symbolic link.")
        return path

    def _load_records(self) -> list[MemoryRecord]:
        path = self._store_path(create_directory=False)
        if not path.exists():
            return []
        if not path.is_file():
            raise RuntimeError(f"Memory store is not a regular file: {path}")

        records: list[MemoryRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # One damaged line should not make every other saved fact
                # unavailable to later sessions.
                continue
            if not isinstance(value, dict):
                continue
            record_id = value.get("id")
            fact = value.get("fact")
            if (
                isinstance(record_id, int)
                and not isinstance(record_id, bool)
                and record_id >= 1
                and isinstance(fact, str)
                and fact.strip()
            ):
                records.append(MemoryRecord(id=record_id, fact=fact))
        return records

    def remember(self, fact: str) -> str:
        """Store one fact verbatim and return a concise confirmation."""

        value = str(fact).strip()
        if not value:
            return "Nothing was stored because the memory fact was empty."

        with _memory_lock:
            records = self._load_records()
            duplicate = next(
                (record for record in records if record.fact == value),
                None,
            )
            if duplicate is not None:
                return f"That fact is already stored as memory #{duplicate.id}."

            record_id = max((record.id for record in records), default=0) + 1
            path = self._store_path(create_directory=True)
            row = {"id": record_id, "fact": value}
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        return f"Stored memory #{record_id}."

    @staticmethod
    def _fallback_matches(
        records: list[MemoryRecord],
        query: str,
    ) -> list[int]:
        """Return record indexes sharing Unicode-aware terms with ``query``."""

        query_terms = _fallback_terms(query)
        if not query_terms:
            return []

        scored: list[tuple[int, int]] = []
        for index, record in enumerate(records):
            overlap = len(query_terms & _fallback_terms(record.fact))
            if overlap:
                scored.append((index, overlap))
        scored.sort(key=lambda item: (-item[1], -records[item[0]].id))
        return [index for index, _score in scored]

    def recall(self, query: str) -> str:
        """Return a compact JSON object containing relevant saved facts."""

        with _memory_lock:
            records = self._load_records()

        if not records:
            return json.dumps(
                {
                    "matches": [],
                    "message": "No memories have been stored yet.",
                },
                ensure_ascii=False,
            )

        normalized_query = str(query).strip()
        if not normalized_query:
            matched_indexes = list(range(len(records)))
        else:
            index = build_bm25([record.fact for record in records])
            # Ask BM25 for every lexical match, then present the small result
            # set newest-first.  This makes an explicit correction visible
            # before the older fact that it supersedes.
            ranked = search_bm25(index, normalized_query, k=len(records))
            matched_indexes = [record_index for record_index, _score in ranked]
            if not matched_indexes:
                matched_indexes = self._fallback_matches(
                    records,
                    normalized_query,
                )

        matched_indexes = sorted(
            set(matched_indexes),
            key=lambda index: records[index].id,
            reverse=True,
        )[:MAX_RECALL_RESULTS]

        matches = [
            {"id": records[index].id, "fact": records[index].fact}
            for index in matched_indexes
        ]
        message = (
            "Relevant memories are listed newest first."
            if matches
            else "No stored memories matched the query."
        )
        return json.dumps(
            {"matches": matches, "message": message},
            ensure_ascii=False,
        )
