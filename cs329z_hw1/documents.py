"""Deterministic search and bounded reading for the company documents.

Search operates over overlapping passages so a long document does not drown
out a short relevant section.  Reading uses the original document text and
returns exact, consecutive source lines.  The two operations deliberately
share one index object but have different output-size limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from cs329z_hw1.bm25 import BM25Index, build_bm25, search_bm25
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import count_tokens
from cs329z_hw1.types import DocWindow, SearchResult


PASSAGE_TARGET_CHARS = 1_400
PASSAGE_OVERLAP_LINES = 3
SNIPPET_MAX_TOKENS = 150
WINDOW_MAX_TOKENS = 1_500

# ``count_tokens`` charges one token per four characters, rounded up.  These
# exact character limits therefore imply the token limits above.
SNIPPET_MAX_CHARS = SNIPPET_MAX_TOKENS * 4
WINDOW_MAX_CHARS = WINDOW_MAX_TOKENS * 4


@dataclass(frozen=True, slots=True)
class Document:
    """The immutable source fields needed by search and reading."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Passage:
    """One contiguous, searchable line range from a source document."""

    doc_id: str
    title: str
    start_line: int
    end_line: int
    text: str


@dataclass(slots=True)
class DocumentIndex:
    """Original documents plus a BM25 index over their passages."""

    documents: dict[str, Document]
    passages: tuple[Passage, ...]
    bm25: BM25Index


def _make_passages(document: Document) -> list[Passage]:
    """Split a document into overlapping, contiguous groups of lines."""

    lines = document.text.splitlines()
    passages: list[Passage] = []
    start = 0

    while start < len(lines):
        end = start
        size = 0

        # Always include at least one complete source line.  Subsequent lines
        # are added while the passage remains near the target size.
        while end < len(lines):
            added = len(lines[end]) + (1 if end > start else 0)
            if end > start and size + added > PASSAGE_TARGET_CHARS:
                break
            size += added
            end += 1

        passage_text = "\n".join(lines[start:end])
        passages.append(
            Passage(
                doc_id=document.doc_id,
                title=document.title,
                start_line=start + 1,
                end_line=end,
                text=passage_text,
            )
        )

        if end == len(lines):
            break

        # Repeating a few complete lines protects facts located at a chunk
        # boundary while ensuring that every iteration advances.
        start = max(start + 1, end - PASSAGE_OVERLAP_LINES)

    return passages


def build_doc_index(docs: list[dict]) -> DocumentIndex:
    """Build an immutable document lookup and a BM25 passage index.

    The input dictionaries are copied into small immutable records.  This
    both avoids mutating caller-owned data and prevents later caller changes
    from silently invalidating the BM25 statistics.
    """

    documents: dict[str, Document] = {}
    passages: list[Passage] = []

    for raw in docs:
        document = Document(
            doc_id=str(raw["doc_id"]),
            title=str(raw["title"]),
            text=str(raw["text"]),
        )
        if document.doc_id in documents:
            raise ValueError(f"Duplicate document id: {document.doc_id!r}.")
        documents[document.doc_id] = document
        passages.extend(_make_passages(document))

    frozen_passages = tuple(passages)
    return DocumentIndex(
        documents=documents,
        passages=frozen_passages,
        bm25=build_bm25([passage.text for passage in frozen_passages]),
    )


def _term_positions(text: str, terms: list[str]) -> list[int]:
    """Find query-token positions using the same token boundaries as BM25."""

    lowered = text.lower()
    positions: list[int] = []
    for term in terms:
        match = re.search(
            rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
            lowered,
        )
        if match is not None:
            positions.append(match.start())
    return positions


def _snippet(passage: Passage, query: str) -> str:
    """Return a short source excerpt centred near the first query match."""

    text = passage.text
    if len(text) <= SNIPPET_MAX_CHARS:
        return text

    terms = list(dict.fromkeys(tokenize(query)))
    positions = _term_positions(text, terms)
    focus = min(positions) if positions else 0

    # Put about one third of the excerpt before the first match, leaving more
    # room after it for the surrounding fact or sentence.
    start = max(0, focus - SNIPPET_MAX_CHARS // 3)
    end = min(len(text), start + SNIPPET_MAX_CHARS)
    start = max(0, end - SNIPPET_MAX_CHARS)

    # Avoid beginning or ending in the middle of an ASCII token.  Moving the
    # boundaries inward preserves the hard character/token limit.
    if start > 0:
        while start < focus and text[start] in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            start += 1
        while start < end and text[start].isspace():
            start += 1
    if end < len(text):
        while end > start and text[end - 1] in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            end -= 1
        while end > start and text[end - 1].isspace():
            end -= 1

    snippet = text[start:end]
    # The passage itself is non-empty for the shipped corpus.  This fallback
    # also keeps the function useful for unusual text made only of whitespace.
    return snippet if snippet.strip() else text[:SNIPPET_MAX_CHARS]


def search_docs(
    index: DocumentIndex,
    query: str,
    k: int = 5,
) -> list[SearchResult]:
    """Return the best matching document passages as compact pointers."""

    if k <= 0:
        return []

    hits = search_bm25(index.bm25, query, k)
    results: list[SearchResult] = []
    for passage_id, _score in hits:
        passage = index.passages[passage_id]
        snippet = _snippet(passage, query)
        # Keep the assertion close to output construction so a future change
        # cannot accidentally turn search into an unbounded document read.
        if count_tokens(snippet) > SNIPPET_MAX_TOKENS:
            raise AssertionError("search_docs produced an oversized snippet")
        results.append(
            {
                "doc_id": passage.doc_id,
                "title": passage.title,
                "snippet": snippet,
                # Extra pointer fields are useful to an agent deciding where
                # to continue with read_doc; the assignment permits them.
                "start_line": passage.start_line,
                "end_line": passage.end_line,
            }
        )
    return results


def read_doc(
    index: DocumentIndex,
    doc_id: str,
    start_line: int = 1,
) -> DocWindow:
    """Read exact source lines beginning at ``start_line`` within the limit."""

    document = index.documents.get(doc_id)
    if document is None:
        raise LookupError(f"Unknown document id: {doc_id!r}.")

    lines = document.text.splitlines()
    total_lines = len(lines)
    if start_line < 1 or start_line > total_lines:
        raise ValueError(
            f"start_line must be between 1 and {total_lines} for {doc_id!r}; "
            f"got {start_line}."
        )

    selected: list[str] = []
    size = 0
    for line in lines[start_line - 1 :]:
        added = len(line) + (1 if selected else 0)
        if selected and size + added > WINDOW_MAX_CHARS:
            break
        if not selected and added > WINDOW_MAX_CHARS:
            raise ValueError(
                f"Line {start_line} of {doc_id!r} exceeds the read window limit."
            )
        selected.append(line)
        size += added

    text = "\n".join(selected)
    if count_tokens(text) > WINDOW_MAX_TOKENS:
        raise AssertionError("read_doc produced an oversized window")

    end_line = start_line + len(selected) - 1
    return {
        "doc_id": document.doc_id,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
        "text": text,
    }
