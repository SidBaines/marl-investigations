"""Per (cell, condition, metric): n, misses, a 95% interval and the gain over a baseline.

Intervals follow the suite's declared pairing:

- ``pair: sample`` (the same tasks in every cell, e.g. HiddenBench scenarios): repeats
  of a sample are one statistical unit. Wilson for a binary metric with one value per
  sample, else a seeded bootstrap over per-sample means. The gain is paired on shared
  samples: bootstrap over per-sample differences, exact McNemar (binary, one value per
  sample) or a sign-flip test otherwise (marli.eval.stats).
- ``pair: none`` (i.i.d. trials, e.g. one-shot games): Wilson (binary) or a bootstrap of
  the mean (shares); the gain is an independent difference of means with a two-sample
  bootstrap interval and a seeded permutation test.

Unparsed values (None) are excluded from n and counted as ``missing``; ``parse_failures``
also counts replies a harness re-asked (FAIRGAME). Harness errors are counted
separately. Statistics never see prompt text.
"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import fmean
from typing import Any

from marli.eval.external.spec import ExternalSuite
from marli.eval.stats import bootstrap_mean, mcnemar_exact, percentile, sign_flip_p, wilson_ci

RESAMPLES = 2000
PERMUTATIONS = 10000


def _by_sample(
    rows: list[dict[str, Any]], metric: str, key: str = "sample"
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in rows:
        value = item["metrics"].get(metric)
        if value is not None:
            grouped[item.get(key) or item["sample"]].append(float(value))
    return dict(grouped)


def _interval(values: dict[str, list[float]], kind: str, pair: str) -> tuple[float, float, str]:
    flat = [value for group in values.values() for value in group]
    singletons = all(len(group) == 1 for group in values.values())
    if kind == "binary" and (pair == "none" or singletons):
        low, high = wilson_ci(sum(flat), len(flat))
        return low, high, "wilson"
    units = flat if pair == "none" else [fmean(group) for group in values.values()]
    estimate = bootstrap_mean(units, resamples=RESAMPLES)
    return estimate.low, estimate.high, "bootstrap" if pair == "none" else "sample_bootstrap"


def _paired_gain(
    cell: dict[str, list[float]], base: dict[str, list[float]], kind: str
) -> dict[str, Any] | None:
    shared = sorted(cell.keys() & base.keys())
    if not shared:
        return None
    differences = [fmean(cell[key]) - fmean(base[key]) for key in shared]
    estimate = bootstrap_mean(differences, resamples=RESAMPLES)
    singletons = all(len(cell[key]) == 1 and len(base[key]) == 1 for key in shared)
    if kind == "binary" and singletons:
        wins = sum(value > 0 for value in differences)
        losses = sum(value < 0 for value in differences)
        test, p = "mcnemar", mcnemar_exact(wins, losses)
    else:
        test, p = "sign_flip", sign_flip_p(differences)
    return {
        "difference": estimate.mean,
        "low": estimate.low,
        "high": estimate.high,
        "p": p,
        "test": test,
        "n_pairs": len(shared),
    }


def _independent_gain(cell: list[float], base: list[float]) -> dict[str, Any] | None:
    if not cell or not base:
        return None
    rng = random.Random(0)
    observed = fmean(cell) - fmean(base)
    draws = [
        fmean(rng.choices(cell, k=len(cell))) - fmean(rng.choices(base, k=len(base)))
        for _ in range(RESAMPLES)
    ]
    pooled = cell + base
    extreme = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pooled)
        if abs(fmean(pooled[: len(cell)]) - fmean(pooled[len(cell) :])) >= abs(observed) - 1e-12:
            extreme += 1
    return {
        "difference": observed,
        "low": percentile(draws, 2.5),
        "high": percentile(draws, 97.5),
        "p": (extreme + 1) / (PERMUTATIONS + 1),
        "test": "permutation",
        "n_pairs": None,
    }


def summarize(
    cells: dict[str, list[dict[str, Any]]], suite: ExternalSuite, baseline: str | None
) -> list[dict[str, Any]]:
    """One result per (cell, condition, metric) present in the rows, in a stable order."""
    results = []
    conditions = sorted({item["condition"] for rows in cells.values() for item in rows})
    for label, rows in cells.items():
        for condition in conditions:
            here = [item for item in rows if item["condition"] == condition]
            if not here:
                continue
            for spec in suite.metrics:
                metric = spec["name"]
                present = [item for item in here if metric in item["metrics"]]
                if not present:
                    continue
                values = _by_sample(present, metric)
                flat = [value for group in values.values() for value in group]
                result: dict[str, Any] = {
                    "cell": label,
                    "condition": condition,
                    "metric": metric,
                    "kind": spec["kind"],
                    "pair": spec["pair"],
                    "n": len(flat),
                    "n_samples": len(values),
                    "missing": sum(item["metrics"][metric] is None for item in present),
                    "parse_failures": sum(item["parse_failures"] for item in here),
                    "errors": sum(item["error"] is not None for item in here),
                    "mean": fmean(flat) if flat else None,
                    "ci": None,
                    "ci_method": None,
                    "gain": None,
                    "vs_control": None,
                }
                if flat:
                    low, high, method = _interval(values, spec["kind"], spec["pair"])
                    result.update(ci=[low, high], ci_method=method)
                if baseline is not None and label != baseline and flat:
                    base_rows = [
                        item
                        for item in cells[baseline]
                        if item["condition"] == condition and metric in item["metrics"]
                    ]
                    base_values = _by_sample(base_rows, metric)
                    if spec["pair"] == "sample":
                        result["gain"] = _paired_gain(values, base_values, spec["kind"])
                    else:
                        result["gain"] = _independent_gain(
                            flat, [v for group in base_values.values() for v in group]
                        )
                if suite.control is not None and condition != suite.control and flat:
                    control_rows = [
                        item
                        for item in rows
                        if item["condition"] == suite.control and metric in item["metrics"]
                    ]
                    result["vs_control"] = _paired_gain(
                        _by_sample(present, metric, "unit"),
                        _by_sample(control_rows, metric, "unit"),
                        spec["kind"],
                    )
                results.append(result)
    return results


def markdown(
    results: list[dict[str, Any]], suite: ExternalSuite, baseline: str | None, notes: list[str]
) -> str:
    def rate(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    lines = [
        f"# {suite.name}",
        "",
        suite.description.strip(),
        "",
        f"Source: {suite.source}. Citation: {suite.citation}",
        "",
        "| Cell | Condition | Metric | n | Unparsed | Errors | Mean [95% CI] | "
        f"Gain vs {baseline or '—'} [95% CI] | p (test) |"
        + (f" vs {suite.control} [95% CI] | p |" if suite.control else ""),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        + (" --- | --- |" if suite.control else ""),
    ]
    for item in results:
        ci = item["ci"]
        mean = f"{rate(item['mean'])} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else rate(item["mean"])
        gain = item["gain"]
        gain_text = (
            f"{gain['difference']:+.3f} [{gain['low']:+.3f}, {gain['high']:+.3f}]" if gain else "—"
        )
        p_text = f"{gain['p']:.3g} ({gain['test']})" if gain else "—"
        line = (
            f"| {item['cell']} | {item['condition']} | {item['metric']} | {item['n']} | "
            f"{item['parse_failures']} | {item['errors']} | {mean} | {gain_text} | {p_text} |"
        )
        if suite.control:
            contrast = item["vs_control"]
            line += (
                f" {contrast['difference']:+.3f} [{contrast['low']:+.3f}, {contrast['high']:+.3f}]"
                f" | {contrast['p']:.3g} |"
                if contrast
                else " — | — |"
            )
        lines.append(line)
    lines.extend(
        [
            "",
            "n counts parsed values. Unparsed counts replies with no readable decision: "
            "excluded from n where the harness keeps them (Inspect games), re-asked where "
            "the harness re-asks (FAIRGAME). Errors counts harness/model failures (excluded). "
            "Intervals: Wilson for binary rates over "
            "independent units, otherwise a 2,000-draw seeded bootstrap (over samples when "
            "tasks repeat across cells). Gains over the baseline cell are paired on shared "
            "samples where the suite pairs them (McNemar or sign-flip p), otherwise an "
            "independent difference (two-sample bootstrap, permutation p).",
        ]
    )
    if suite.control:
        lines.extend(
            [
                "",
                f"vs {suite.control}: within each cell, the condition minus the {suite.control} "
                "condition, paired on the same tasks (bootstrap over tasks; McNemar or "
                "sign-flip p).",
            ]
        )
    if notes:
        lines.extend(["", "Notes:", *(f"- {note}" for note in notes)])
    return "\n".join(lines) + "\n"
