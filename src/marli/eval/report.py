"""Reports expose sample sizes and compute beside accuracy, preserving task pairing.

Standalone and grid reports use the same labelled inputs and task-level statistics.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any, ClassVar

from marli.config import input_field
from marli.errors import ConfigError
from marli.eval.score import Scores
from marli.eval.stats import (
    avg_at_k,
    bootstrap_mean,
    maj_at_k,
    paired_comparison,
    percentile,
    wilson_ci,
)
from marli.handles import Handle, InputRef, atomic_write_text, register_handle
from marli.rundir import RunDir
from marli.verbs import input_paths


@dataclass
class ReportConfig:
    scores: str | None = input_field(
        None, help="Scores manifest/dir or label=path entries joined by os.pathsep"
    )
    baseline: str | None = None
    group_by: list[str] = field(default_factory=lambda: ["protocol", "policy", "taskset"])

    def __post_init__(self) -> None:
        if len(set(self.group_by)) != len(self.group_by):
            raise ConfigError("group_by fields must be unique")


@register_handle
@dataclass(frozen=True)
class Report(Handle):
    KIND: ClassVar[str] = "report"
    MANIFEST: ClassVar[str] = "report.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("results", "markdown")

    results: str
    markdown: str
    n_cells: int

    def summary(self) -> dict[str, Any]:
        return {"n_cells": self.n_cells}


def build_report(sources: dict[str, Scores], cfg: ReportConfig, run: RunDir) -> Report:
    """Aggregate labelled sources; empty labels derive their name from group_by values."""
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups: dict[str, dict[str, Any]] = {}
    costs: dict[str, float] = defaultdict(float)
    for label, source in sources.items():
        source_cells: dict[str, int] = defaultdict(int)
        with source.file("rows").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                missing = set(cfg.group_by) - row.keys()
                if missing:
                    raise ConfigError(f"unknown group_by fields: {sorted(missing)}")
                group = {key: row[key] for key in cfg.group_by}
                cell = (
                    label or " | ".join(f"{key}={value}" for key, value in group.items()) or "all"
                )
                if cell in groups and groups[cell] != group:
                    raise ConfigError(f"label {cell!r} contains multiple group_by cells")
                groups[cell] = group
                cells[cell].append(row)
                source_cells[cell] += 1
                costs[cell] += row.get("cost_usd", 0.0)
        # Rollout spend includes failed attempts absent from the final score rows.
        if len(source_cells) == 1 and "cost_usd" in source.meta:
            cell = next(iter(source_cells))
            costs[cell] += source.meta["cost_usd"] - sum(
                row.get("cost_usd", 0.0) for row in cells[cell][-source_cells[cell] :]
            )
    if cfg.baseline is not None and cfg.baseline not in cells:
        raise ConfigError(f"unknown baseline {cfg.baseline!r}; cell labels: {sorted(cells)}")
    task_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for label, rows in cells.items():
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[row["task_id"]].append(row)
        for episodes in by_task.values():
            episodes.sort(key=lambda row: (row["episode_idx"], row["episode_id"]))
        task_rows[label] = by_task

    results = []
    for label, rows in sorted(cells.items()):
        by_task = task_rows[label]
        correct = sum(row["correct"] for row in rows)
        sizes = sorted({len(episodes) for episodes in by_task.values()})
        repeated = max(sizes) > 1
        means = [fmean(row["correct"] for row in eps) for _, eps in sorted(by_task.items())]
        accuracy = fmean(means) if repeated else correct / len(rows)
        if repeated:
            estimate = bootstrap_mean(means)
            ci = (estimate.low, estimate.high)
        else:
            ci = wilson_ci(correct, len(rows))
        ok_rows = [row for row in rows if row["ok"]]
        n_failed = len(rows) - len(ok_rows)
        if n_failed:
            warnings.warn(
                f"{label}: {n_failed} failed episodes; accuracy_ok excludes them", stacklevel=2
            )
        result: dict[str, Any] = {
            "label": label,
            "group": groups[label],
            "n_tasks": len(by_task),
            "n_episodes": len(rows),
            "accuracy": accuracy,
            "accuracy_ok": fmean(row["correct"] for row in ok_rows) if ok_rows else None,
            "accuracy_ci_method": "task_bootstrap" if repeated else "wilson",
            "cost_usd": costs[label],
            "accuracy_ci": list(ci),
            "episodes_per_task": sizes,
            "avg_at_k": None,
            "maj_at_k": None,
            "oracle_any": fmean(row["oracle_any"] for row in rows),
            "answered_rate": fmean(row["answered"] for row in rows),
            "n_failed": n_failed,
            "compute": {},
            "paired_lift": None,
        }
        if max(sizes) > 1:
            result["avg_at_k"] = fmean(
                avg_at_k([row["correct"] for row in episodes]) for episodes in by_task.values()
            )
            if all("answer_group" in row for row in rows):
                result["maj_at_k"] = fmean(
                    maj_at_k(
                        [row["answer_group"] for row in episodes],
                        [row["correct"] for row in episodes],
                    )
                    for episodes in by_task.values()
                )
        for metric in (
            "total_gen",
            "cp_tokens",
            "calls",
            "peak_ctx",
            "total_uncached",
            "api_gen_tokens",
            "api_prompt_tokens",
        ):
            values = [row[metric] for row in rows if metric in row]
            if values:
                result["compute"][metric] = {
                    "mean": fmean(values),
                    "p50": percentile(values, 50),
                    "p90": percentile(values, 90),
                }
        if cfg.baseline is not None:
            baseline_rows = cells[cfg.baseline]
            harness = {row["harness"] for row in rows}
            baseline_harness = {row["harness"] for row in baseline_rows}
            if len(harness) != 1 or harness != baseline_harness:
                result["comparison_note"] = (
                    "different taskset or environment or regrade setting; no paired comparison"
                )
            elif not (by_task.keys() & task_rows[cfg.baseline].keys()):
                result["comparison_note"] = "no shared tasks; no paired comparison"
            else:
                result["paired_lift"] = asdict(
                    paired_comparison(
                        {task: [row["correct"] for row in eps] for task, eps in by_task.items()},
                        {
                            task: [row["correct"] for row in eps]
                            for task, eps in task_rows[cfg.baseline].items()
                        },
                    )
                )
        results.append(result)

    atomic_write_text(
        run.path("results.jsonl"),
        "".join(json.dumps(result, sort_keys=True) + "\n" for result in results),
    )
    atomic_write_text(run.path("RESULTS.md"), _markdown(results, cfg.baseline))
    return Report(
        root=run.out,
        inputs=tuple(InputRef.of(source) for source in sources.values()),
        results="results.jsonl",
        markdown="RESULTS.md",
        n_cells=len(results),
        meta={
            "warnings": [
                f"{row['label']}: {row['n_failed']} failed episodes; accuracy_ok excludes them"
                for row in results
                if row["n_failed"]
            ]
        },
    )


def _markdown(results: list[dict[str, Any]], baseline: str | None) -> str:
    lines = [
        "# Evaluation results",
        "",
        "| Cell | Tasks | Episodes | Accuracy [95% CI] | avg@k | maj@k | Oracle | Answered | "
        "Failed | Accuracy ok | Cost USD | Total gen mean/p50/p90 | "
        "CP tokens mean/p50/p90 | Calls mean/p50/p90 | "
        "Peak ctx mean/p50/p90 | Total uncached mean/p50/p90 | "
        "Paired lift [95% CI] | Paired n | Paired test | p |",
        "| " + " | ".join(["---"] * 20) + " |",
    ]

    def rate(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    for row in results:
        ci = row["accuracy_ci"]
        lift = row["paired_lift"]
        values = [
            row["label"].replace("|", "\\|").replace("\n", " "),
            str(row["n_tasks"]),
            str(row["n_episodes"]),
            f"{row['accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
            rate(row["avg_at_k"]),
            rate(row["maj_at_k"]),
            rate(row["oracle_any"]),
            rate(row["answered_rate"]),
            str(row["n_failed"]),
            rate(row["accuracy_ok"]),
            f"{row['cost_usd']:.6f}",
        ]
        for metric in ("total_gen", "cp_tokens", "calls", "peak_ctx", "total_uncached"):
            stats = row["compute"].get(metric)
            values.append(
                "/".join(f"{stats[key]:.1f}" for key in ("mean", "p50", "p90")) if stats else "—"
            )
        values.extend(
            [
                f"{lift['difference']:+.3f} [{lift['low']:+.3f}, {lift['high']:+.3f}]"
                if lift
                else "—",
                str(lift["n_tasks"]) if lift else "—",
                lift["test"] if lift else "—",
                f"{(lift['mcnemar_p'] if lift['test'] == 'mcnemar' else lift['permutation_p']):.4g}"
                if lift
                else "—",
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Comparisons are within-harness and should be read against compute: compare both "
            "total generated tokens and critical-path tokens, plus calls and context. Token counts "
            "from different tokenizers are not comparable; API token usage is separate "
            "in results.jsonl.",
            "",
            "Accuracy uses Wilson intervals for G=1; for G>1 it averages per-task means "
            "with 2,000 seeded task bootstrap resamples for its interval. "
            "avg@k and maj@k average tasks equally; k is each task's recorded episode count. "
            "maj@k votes over verifier-equivalent answers; ties use the earliest episode.",
            "",
            "Paired lift averages per-task accuracy differences on shared tasks with 2,000 "
            "seeded task bootstrap resamples. G=1 uses exact McNemar; G>1 uses a sign-flip "
            "permutation test on per-task differences (exact for up to 20 nonzero tasks, "
            "otherwise 10,000 seeded draws).",
        ]
    )
    lines.extend(
        [
            "",
            "## Accuracy vs compute",
            "",
            "| Cell | Accuracy [95% CI] | CI method | Mean total_gen | Mean cp_tokens |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in results:
        low, high = row["accuracy_ci"]
        compute = row["compute"]
        label = row["label"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {label} | {row['accuracy']:.3f} [{low:.3f}, {high:.3f}] | "
            f"{row['accuracy_ci_method']} | {rate(compute.get('total_gen', {}).get('mean'))} | "
            f"{rate(compute.get('cp_tokens', {}).get('mean'))} |"
        )
    for row in results:
        if row["n_failed"]:
            lines.extend(["", f"Warning: {row['label']} has {row['n_failed']} failed episodes."])
    if baseline is not None:
        base = next(row for row in results if row["label"] == baseline)
        base_gen = base["compute"].get("total_gen", {}).get("mean")
        unmatched = [
            row["label"]
            for row in results
            if base_gen is not None
            and "total_gen" in row["compute"]
            and abs(row["compute"]["total_gen"]["mean"] - base_gen) > 0.1 * base_gen
        ]
        if unmatched:
            lines.extend(
                [
                    "",
                    "Not compute-matched (>10% mean total_gen difference from "
                    f"baseline): {', '.join(unmatched)}.",
                ]
            )
    if baseline is not None:
        lines.extend(["", f"Baseline: {baseline}."])
    for row in results:
        if "comparison_note" in row:
            lines.extend(["", f"{row['label']}: {row['comparison_note']}."])
    return "\n".join(lines) + "\n"


async def report(cfg: ReportConfig, run: RunDir) -> Report:
    if cfg.scores is None:
        raise ConfigError("report requires Scores manifests")
    return build_report(
        {label: Scores.load(path) for label, path in input_paths(cfg.scores).items()}, cfg, run
    )
