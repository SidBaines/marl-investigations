"""Find problems that repeat a reference set's problems, including rewordings, without any text.

Exact-text decontamination (``data build exclude=``) misses the same problem copied into another
dataset subset with a different wrapper, layout or wording. These checks catch such copies:

- **Text.** Lines common across a pool (fixed prompt wrappers, section headers, sample-data lines)
  are dropped first (:func:`common_lines`, :func:`strip_lines`), then 5-word shingles are compared
  by Jaccard similarity and by containment (shared / smaller set), which also catches a copy
  embedded in a longer statement.
- **Tests.** Each input/output pair is reduced to case-folded tokens without whitespace, quotes or
  JSON brackets (:func:`case_key`), so stdin and argument-list copies of a test agree. Copies
  share pairs; unrelated problems share only trivial ones, which :func:`shared_cases` discounts by
  how many pool problems contain them.

Everything here is pure and dependency-free; callers bring the problems and, for meaning-based
checks, their own embeddings. Results are ids and numbers only, never problem text.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

_WORD = re.compile(r"\w+")
_TOKEN = re.compile(r"[^\s\[\]{}(),;\"']+")


def _hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def words(text: str) -> list[str]:
    """Case-folded word tokens; punctuation, markup and layout carry no identity."""
    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def normalise_line(line: str) -> str:
    return " ".join(words(line))


def common_lines(prompts: Iterable[str], min_count: int) -> frozenset[str]:
    """Normalised lines that occur in at least ``min_count`` prompts (each prompt counts once)."""
    if type(min_count) is not int or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    counts: Counter[str] = Counter()
    for prompt in prompts:
        counts.update({normalise_line(line) for line in prompt.splitlines()})
    return frozenset(line for line, count in counts.items() if count >= min_count)


def strip_lines(prompt: str, common: frozenset[str]) -> str:
    """The prompt without blank lines and without lines whose normalised form is common."""
    kept = []
    for line in prompt.splitlines():
        normal = normalise_line(line)
        if normal and normal not in common:
            kept.append(line)
    return "\n".join(kept)


def shingles(text: str, n: int = 5) -> frozenset[int]:
    """Hashes of every run of ``n`` consecutive words (one hash for shorter texts)."""
    tokens = words(text)
    if not tokens:
        return frozenset()
    if len(tokens) < n:
        return frozenset({_hash(" ".join(tokens))})
    return frozenset(_hash(" ".join(tokens[i : i + n])) for i in range(len(tokens) - n + 1))


def _case_tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def case_key(input: str, output: str) -> int:
    return _hash(" ".join(_case_tokens(input)) + "\x1f" + " ".join(_case_tokens(output)))


def case_size(input: str, output: str) -> int:
    """Characters left in a pair after normalisation; short pairs coincide by chance."""
    return sum(map(len, _case_tokens(input))) + sum(map(len, _case_tokens(output)))


def case_keys(tests: Sequence[Mapping[str, str]]) -> frozenset[int]:
    """Keys of all of a problem's test pairs (pass public and hidden tests together)."""
    return frozenset(case_key(test["input"], test["output"]) for test in tests)


def document_frequency(sets: Iterable[frozenset[int]]) -> Counter[int]:
    """In how many of the sets each item occurs."""
    counts: Counter[int] = Counter()
    for items in sets:
        counts.update(items)
    return counts


def rare(items: frozenset[int], frequency: Mapping[int, int], max_df: int) -> frozenset[int]:
    """The items found in at most ``max_df`` sets of the pool ``frequency`` was counted on."""
    return frozenset(item for item in items if frequency.get(item, 0) <= max_df)


@dataclass(frozen=True)
class TextMatch:
    other: str
    jaccard: float
    containment: float
    shared: int


def text_matches(
    queries: Mapping[str, frozenset[int]],
    pool: Mapping[str, frozenset[int]],
    *,
    max_postings: int | None = None,
    order: Mapping[str, int] | None = None,
) -> dict[str, list[TextMatch]]:
    """Every pool problem sharing a shingle with each query, best Jaccard first.

    ``max_postings`` ignores shingles found in more pool problems than that (stock phrases).
    With ``order`` (queries and pool are the same set), a query is compared only with problems
    that come before it, so each duplicate pair is reported once, against its earlier copy.
    """
    index: dict[int, list[str]] = defaultdict(list)
    for key, items in pool.items():
        for item in items:
            index[item].append(key)
    out: dict[str, list[TextMatch]] = {}
    for query, items in queries.items():
        counts: Counter[str] = Counter()
        for item in items:
            postings = index.get(item, ())
            if max_postings is not None and len(postings) > max_postings:
                continue
            counts.update(postings)
        matches = []
        for other, shared in counts.items():
            if other == query or (order is not None and order[other] >= order[query]):
                continue
            a, b = len(items), len(pool[other])
            matches.append(TextMatch(other, shared / (a + b - shared), shared / min(a, b), shared))
        out[query] = sorted(matches, key=lambda m: (-m.jaccard, -m.containment, m.other))
    return out


@dataclass(frozen=True)
class CaseMatch:
    other: str
    shared: int  # test pairs in common
    distinctive: int  # of which found in at most max_df pool problems
    long: int  # of which also at least min_chars long after normalisation
    contained: bool  # one problem's whole suite is among the other's pairs


def shared_cases(
    queries: Mapping[str, frozenset[int]],
    pool: Mapping[str, frozenset[int]],
    frequency: Mapping[int, int],
    *,
    max_df: int,
    sizes: Mapping[int, int] | None = None,
    min_chars: int = 0,
    order: Mapping[str, int] | None = None,
) -> dict[str, list[CaseMatch]]:
    """Pool problems sharing test pairs with each query, most distinctive first.

    ``frequency`` counts, per pair key, the problems of the whole source pool that contain it;
    a pair found in more than ``max_df`` of them (e.g. input 1, output 1) is not distinctive.
    ``sizes`` (pair key -> :func:`case_size`) counts the distinctive pairs of at least
    ``min_chars`` characters separately: short pairs can coincide in unrelated problems.
    """
    index: dict[int, list[str]] = defaultdict(list)
    for key, items in pool.items():
        for item in items:
            index[item].append(key)
    out: dict[str, list[CaseMatch]] = {}
    for query, items in queries.items():
        shared: Counter[str] = Counter()
        distinctive: Counter[str] = Counter()
        long: Counter[str] = Counter()
        for item in items:
            for other in index.get(item, ()):
                shared[other] += 1
                if frequency.get(item, 0) <= max_df:
                    distinctive[other] += 1
                    if sizes is not None and sizes.get(item, 0) >= min_chars:
                        long[other] += 1
        matches = []
        for other, count in shared.items():
            if other == query or (order is not None and order[other] >= order[query]):
                continue
            contained = count == min(len(items), len(pool[other]))
            matches.append(CaseMatch(other, count, distinctive[other], long[other], contained))
        out[query] = sorted(matches, key=lambda m: (-m.distinctive, -m.shared, m.other))
    return out


@dataclass(frozen=True)
class Rules:
    """When a problem counts as a repeat of another. Defaults: sacrifice-relay exp2_eval."""

    jaccard: float = 0.05  # text: Jaccard of rare shingles
    containment: float = 0.1  # text: shared / smaller set of rare shingles ...
    containment_min: int = 15  # ... when both have at least this many
    contained_min: int = 2  # tests: a whole suite inside the other's, with at least this many pairs
    long_min: int = 1  # tests: rare pairs of at least min_chars characters
    distinctive_min: int = 3  # tests: rare pairs of any length

    def text(self, matches: Sequence[TextMatch], sizes: Mapping[str, int], query: str) -> bool:
        return any(
            m.jaccard >= self.jaccard
            or (
                m.containment >= self.containment
                and min(sizes[query], sizes[m.other]) >= self.containment_min
            )
            for m in matches
        )

    def cases(self, matches: Sequence[CaseMatch]) -> bool:
        return any(
            (m.contained and m.shared >= self.contained_min)
            or m.long >= self.long_min
            or m.distinctive >= self.distinctive_min
            for m in matches
        )


def histogram(values: Iterable[float], edges: Sequence[float]) -> list[tuple[str, int]]:
    """Counts per half-open bin [edges[i], edges[i + 1]); the last bin is closed."""
    values = list(values)
    bins = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:], strict=False)):
        last = i == len(edges) - 2
        count = sum(lo <= v < hi or (last and v == hi) for v in values)
        bins.append((f"[{lo:g}, {hi:g}{']' if last else ')'}", count))
    return bins
