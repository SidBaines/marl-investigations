"""Task-level resampling preserves pairing and avoids treating repeats as new tasks."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist, fmean


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class PairedComparison:
    n_tasks: int
    difference: float
    low: float
    high: float
    mcnemar_p: float | None
    wins: int
    losses: int
    permutation_p: float | None = None
    test: str = "mcnemar"


def percentile(values: Sequence[float], q: float) -> float:
    """Linearly interpolate the sorted sample at a percentile in [0, 100]."""
    if not values or not 0 <= q <= 100:
        raise ValueError("percentile needs values and q in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def wilson_ci(successes: float, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval; an empty sample conveys no information: [0, 1]."""
    if type(n) is not int or n < 0 or not 0 <= successes <= n or not 0 < confidence < 1:
        raise ValueError("invalid proportion or confidence")
    if n == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - width), min(1.0, center + width)


def bootstrap_mean(
    values: Sequence[float], *, seed: int = 0, resamples: int = 2000, confidence: float = 0.95
) -> Estimate:
    """Percentile bootstrap of the mean, using only a local seeded RNG."""
    if not values or resamples < 1 or not 0 < confidence < 1:
        raise ValueError("bootstrap needs values, positive resamples, and confidence in (0, 1)")
    rng = random.Random(seed)
    means = [fmean(rng.choices(values, k=len(values))) for _ in range(resamples)]
    tail = 100 * (1 - confidence) / 2
    return Estimate(fmean(values), percentile(means, tail), percentile(means, 100 - tail))


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact binomial test of the discordant counts (no continuity correction)."""
    if any(type(value) is not int or value < 0 for value in (wins, losses)):
        raise ValueError("discordant counts must be non-negative integers")
    n = wins + losses
    if n == 0:
        return 1.0
    # Sum exact integers before the final float division, avoiding intermediate underflow.
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, (2 * tail) / (1 << n))


def paired_comparison(
    cell: Mapping[str, Sequence[float]],
    baseline: Mapping[str, Sequence[float]],
    *,
    seed: int = 0,
    resamples: int = 2000,
) -> PairedComparison:
    """Compare per-task means, retaining repeats as one statistical unit.

    G=1 uses exact McNemar. Repeated episodes use a two-sided sign-flip
    test on per-task differences, without thresholding or majority ties.
    """
    shared = sorted(cell.keys() & baseline.keys())
    if not shared:
        raise ValueError("paired comparison needs at least one shared task")
    a = [avg_at_k(cell[task]) for task in shared]
    b = [avg_at_k(baseline[task]) for task in shared]
    estimate = bootstrap_mean(
        [x - y for x, y in zip(a, b, strict=True)], seed=seed, resamples=resamples
    )
    differences = [x - y for x, y in zip(a, b, strict=True)]
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    repeated = any(len(cell[task]) > 1 or len(baseline[task]) > 1 for task in shared)
    return PairedComparison(
        len(shared),
        estimate.mean,
        estimate.low,
        estimate.high,
        None if repeated else mcnemar_exact(wins, losses),
        wins,
        losses,
        sign_flip_p(differences, seed=seed) if repeated else None,
        "sign_flip" if repeated else "mcnemar",
    )


def sign_flip_p(differences: Sequence[float], *, seed: int = 0) -> float:
    """Two-sided task sign-flip test; enumerate ≤20 nonzeros, else 10,000 draws."""
    nonzero = [value for value in differences if value != 0]
    observed = abs(math.fsum(nonzero))
    threshold = observed - 1e-12
    if len(nonzero) <= 20:
        sums = [0.0]
        for value in nonzero:
            sums = [total + sign * value for total in sums for sign in (-1, 1)]
        return sum(abs(total) >= threshold for total in sums) / len(sums)
    rng = random.Random(seed)
    extreme = sum(
        abs(math.fsum(value * rng.choice((-1, 1)) for value in nonzero)) >= threshold
        for _ in range(10000)
    )
    # Include the observed permutation to avoid zero Monte Carlo p-values.
    return (extreme + 1) / 10001


def avg_at_k(correct: Sequence[float], k: int | None = None) -> float:
    """Per-task average correctness among the first k episodes (all if k is None)."""
    k = len(correct) if k is None else k
    if type(k) is not int or not 1 <= k <= len(correct):
        raise ValueError("k must be between 1 and the episode count")
    if any(not 0 <= value <= 1 for value in correct[:k]):
        raise ValueError("correctness must lie in [0, 1]")
    return fmean(correct[:k])


def maj_at_k(
    answers: Sequence[Hashable | None], correct: Sequence[float], k: int | None = None
) -> float:
    """Grade the plurality answer class; ties use the earliest episode, abstentions do not vote.

    The caller supplies verifier-equivalence keys, never raw answer strings.
    """
    if len(answers) != len(correct):
        raise ValueError("answers and correctness must align")
    k = len(answers) if k is None else k
    avg_at_k(correct, k)
    counts = Counter(answer for answer in answers[:k] if answer is not None)
    if not counts:
        return 0.0
    winner = max(counts, key=counts.__getitem__)
    return correct[next(i for i, answer in enumerate(answers[:k]) if answer == winner)]
