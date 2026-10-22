"""A small BM25 retriever backed by an inverted index.

The index deliberately stores only corpus statistics.  It does not keep the
original document strings: callers already own those documents and can use the
integer document ids returned by :func:`search_bm25` to retrieve them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import math

from cs329z_hw1.tokenizer import tokenize


K1 = 1.5
B = 0.75

Posting = tuple[int, int]


@dataclass(slots=True)
class BM25Index:
    """Statistics needed to score queries against a document collection.

    ``postings[term]`` contains only documents in which ``term`` occurs.  The
    precomputed length normalization lets search avoid revisiting the corpus
    or recalculating a document-only part of the BM25 formula.
    """

    postings: dict[str, list[Posting]]
    length_normalizations: list[float]
    num_docs: int
    average_doc_length: float


def build_bm25(docs: list[str]) -> BM25Index:
    """Build a BM25 inverted index in one pass over ``docs``.

    Document ids are their positions in ``docs``.  Empty documents therefore
    still contribute to the document count and average length, even though
    they add no postings.
    """

    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: list[int] = []
    total_tokens = 0

    for doc_id, text in enumerate(docs):
        tokens = tokenize(text)
        doc_length = len(tokens)
        doc_lengths.append(doc_length)
        total_tokens += doc_length

        # Counting within one document gives each posting its term frequency
        # while ensuring that a document appears only once per term.
        for term, frequency in Counter(tokens).items():
            postings[term].append((doc_id, frequency))

    num_docs = len(docs)
    average_doc_length = total_tokens / num_docs if num_docs else 0.0

    if average_doc_length:
        length_normalizations = [
            K1 * (1.0 - B + B * doc_length / average_doc_length)
            for doc_length in doc_lengths
        ]
    else:
        # If every document is empty, no query can match a posting.  Keeping a
        # finite value here nevertheless makes the index well-formed.
        length_normalizations = [K1 * (1.0 - B)] * num_docs

    return BM25Index(
        postings=dict(postings),
        length_normalizations=length_normalizations,
        num_docs=num_docs,
        average_doc_length=average_doc_length,
    )


def search_bm25(index: BM25Index, query: str, k: int) -> list[tuple[int, float]]:
    """Return the top-``k`` documents for ``query`` using Lucene BM25 IDF.

    Only postings belonging to distinct query terms are visited.  Documents
    that share no term with the query are never added to ``scores`` and thus
    can never appear in the result.
    """

    if k <= 0:
        return []

    # dict.fromkeys removes duplicates without making the accumulation order
    # depend on hash randomization.  Repeated query terms count only once.
    query_terms = dict.fromkeys(tokenize(query))
    if not query_terms:
        return []

    scores: dict[int, float] = {}
    for term in query_terms:
        term_postings = index.postings.get(term)
        if not term_postings:
            continue

        document_frequency = len(term_postings)
        idf = math.log1p(
            (index.num_docs - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

        for doc_id, term_frequency in term_postings:
            contribution = (
                idf
                * term_frequency
                * (K1 + 1.0)
                / (term_frequency + index.length_normalizations[doc_id])
            )
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution

    if not scores:
        return []

    # nsmallest with (-score, doc_id) produces descending scores, resolves
    # ties by lower document id, and avoids sorting every match when k is small.
    return heapq.nsmallest(
        k,
        scores.items(),
        key=lambda pair: (-pair[1], pair[0]),
    )
