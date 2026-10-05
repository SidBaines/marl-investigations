"""How the held-out candidates overlap the problems experiments 1 and 2 used (counts only, no text).

Compares a held-out TaskSet with the used one (experiment 1's 800 candidates; their training pool
is the subset in ../out/pool when that directory exists) by problem id within its subset, by
normalised problem text (data build's exclusion rule), by shared word runs of several lengths, and
by the largest 5-word-shingle Jaccard similarity to any used problem (near-duplicates: the same
problem reformatted in another DeepCoder subset).

Usage: python overlap.py <used taskset.json> <held-out taskset.json>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from marli.data.build import _ngrams, _normalize


def load(manifest: Path) -> list[tuple[str, str | None, str]]:
    meta = json.loads(manifest.read_text())
    rows = []
    with open(manifest.parent / meta["tasks"]) as stream:
        for line in stream:
            task = json.loads(line)
            rows.append((task["task_id"], task["meta"].get("subset"), _normalize(task["prompt"])))
    return rows


def max_jaccard(shingles: set, others: list[set]) -> float:
    best = 0.0
    for other in others:
        if shingles and other:
            best = max(best, len(shingles & other) / len(shingles | other))
    return best


def main() -> None:
    used_path, held_path = map(Path, sys.argv[1:3])
    used, held = load(used_path), load(held_path)
    pool_path = used_path.parent.parent / "pool" / "taskset.json"
    trained_ids = {task_id for task_id, _, _ in load(pool_path)} if pool_path.exists() else set()
    trained = [text for task_id, _, text in used if task_id in trained_ids]
    print(f"used: {len(used)} problems {dict(Counter(s for _, s, _ in used))}; "
          f"of which trained (pool): {len(trained)}")
    print(f"held out: {len(held)} problems {dict(Counter(s for _, s, _ in held))}")

    def identity(task_id: str) -> str:  # subset/row, without the source name
        return task_id.split("/", 1)[1]

    same_row = {identity(t) for t, _, _ in used} & {identity(t) for t, _, _ in held}
    print("same subset and row:", len(same_row))
    print("same normalised text:", len({p for *_, p in used} & {p for *_, p in held}))
    for n in (8, 13, 20, 30, 50):
        grams = set().union(*(_ngrams(text, n) for *_, text in used))
        grams_trained = set().union(*(_ngrams(text, n) for text in trained)) if trained else set()
        hits = sum(not grams.isdisjoint(_ngrams(text, n)) for *_, text in held)
        hits_trained = sum(not grams_trained.isdisjoint(_ngrams(text, n)) for *_, text in held)
        print(f"share a {n}-word run with a used problem: {hits}; "
              f"with a trained one: {hits_trained}")
    used_sh = [_ngrams(text, 5) for *_, text in used]
    trained_sh = [_ngrams(text, 5) for text in trained]
    best = [max_jaccard(_ngrams(text, 5), used_sh) for *_, text in held]
    best_trained = [max_jaccard(_ngrams(text, 5), trained_sh) for *_, text in held]
    for threshold in (0.3, 0.5, 0.7, 0.9):
        print(f"5-word-shingle Jaccard >= {threshold} to a used problem: "
              f"{sum(j >= threshold for j in best)}; to a trained one: "
              f"{sum(j >= threshold for j in best_trained)}")


if __name__ == "__main__":
    main()
