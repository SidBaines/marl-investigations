"""Reports expose sample sizes and compute beside accuracy, preserving task pairing.

A standalone report consumes one Scores handle (one top-level input path).
Grid combines several labelled Scores handles through the same report builder.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any, ClassVar

from marli.config import input_field
from marli.errors import ConfigError
from marli.eval.score import Scores
from marli.eval.stats import avg_at_k, maj_at_k, paired_comparison, percentile, wilson_ci
from marli.handles import Handle, InputRef, atomic_write_text, register_handle
from marli.rundir import RunDir


@dataclass
class ReportConfig:
    scores: str | None = input_field(
        None, help="One Scores manifest or dir; use eval grid to combine labelled cells"
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
    for label, source in sources.items():
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
        ci = wilson_ci(correct, len(rows))
        sizes = sorted({len(episodes) for episodes in by_task.values()})
        result: dict[str, Any] = {
            "label": label,
            "group": groups[label],
            "n_tasks": len(by_task),
            "n_episodes": len(rows),
            "accuracy": correct / len(rows),
            "accuracy_ci": list(ci),
            "episodes_per_task": sizes,
            "avg_at_k": None,
            "maj_at_k": None,
            "oracle_any": fmean(row["oracle_any"] for row in rows),
            "answered_rate": fmean(row["answered"] for row in rows),
            "n_failed": sum(not row["ok"] for row in rows),
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
                result["comparison_note"] = "different taskset or environment; no paired comparison"
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

    done = run.done_keys("results.jsonl", "label")
    run.path("results.jsonl").touch(exist_ok=True)
    for result in results:
        if result["label"] not in done:
            run.append_row("results.jsonl", result)
    atomic_write_text(run.path("RESULTS.md"), _markdown(results, cfg.baseline))
    return Report(
        root=run.out,
        inputs=tuple(InputRef.of(source) for source in sources.values()),
        results="results.jsonl",
        markdown="RESULTS.md",
        n_cells=len(results),
    )


def _markdown(results: list[dict[str, Any]], baseline: str | None) -> str:
    lines = [
        "# Evaluation results",
        "",
        "| Cell | Tasks | Episodes | Accuracy [95% CI] | avg@k | maj@k | Oracle | Answered | "
        "Failed | Total gen mean/p50/p90 | CP tokens mean/p50/p90 | Calls mean/p50/p90 | "
        "Peak ctx mean/p50/p90 | Paired lift [95% CI] | Paired n | McNemar p |",
        "| " + " | ".join(["---"] * 16) + " |",
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
        ]
        for metric in ("total_gen", "cp_tokens", "calls", "peak_ctx"):
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
                f"{lift['mcnemar_p']:.4g}" if lift else "—",
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
            "Accuracy and Wilson intervals use episodes (repeats are not independent tasks). "
            "avg@k and maj@k average tasks equally; k is each task's recorded episode count. "
            "maj@k votes over verifier-equivalent answers; ties use the earliest episode.",
            "",
            "Paired lift averages per-task accuracy differences on shared tasks with 2,000 "
            "seeded task bootstrap resamples. McNemar uses strict majority-correct per task; "
            "ties count as not majority-correct.",
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
        raise ConfigError("report requires one Scores manifest; use grid for multiple cells")
    return build_report({"": Scores.load(cfg.scores)}, cfg, run)
