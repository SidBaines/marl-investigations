"""Reports expose the paired denominator and refuse cross-harness lift claims."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_eval_rollout import json_rows

from marli.cli.main import main
from marli.eval.report import Report, ReportConfig
from marli.eval.score import Scores
from marli.rundir import RunStatus
from marli.verbs import run_verb


def make_scores(root: Path) -> Scores:
    root.mkdir()
    rows: list[dict[str, Any]] = []
    for cell in ("base", "candidate"):
        for index in range(4):
            for episode_idx in range(2):
                correct = int(cell == "candidate")
                rows.append(
                    {
                        "task_id": f"t{index}",
                        "episode_id": f"{cell}/{index}/{episode_idx}",
                        "episode_idx": episode_idx,
                        "group_id": f"{cell}/{index}",
                        "protocol": cell,
                        "policy": "scripted",
                        "taskset": "tasks",
                        "harness": "arithmetic",
                        "correct": correct,
                        "answered": True,
                        "oracle_any": True,
                        "own_correct": {"solver": correct},
                        "answer_group": 0,
                        "ok": index != 0,
                        "total_gen": 20 if cell == "candidate" else 10,
                        "cp_tokens": 10,
                        "calls": 2 if cell == "candidate" else 1,
                        "peak_ctx": 50,
                    }
                )
    (root / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = Scores(root=root, rows="scores.jsonl", n=len(rows))
    source.save()
    return source


async def test_report_table_compute_paired_lift_and_idempotence(tmp_path: Path) -> None:
    source = make_scores(tmp_path / "scores")
    cfg = ReportConfig(str(source.manifest_path), baseline="protocol=base", group_by=["protocol"])
    result = await run_verb("eval report", cfg, out=tmp_path / "report")
    rows = json_rows(result.handle.file("results"))
    assert len(rows) == 2
    base, candidate = rows
    assert base["accuracy"] == 0 and base["paired_lift"]["difference"] == 0
    assert candidate["accuracy"] == 1 and candidate["n_tasks"] == 4
    assert candidate["n_episodes"] == 8 and candidate["n_failed"] == 2
    assert candidate["avg_at_k"] == candidate["maj_at_k"] == 1
    assert candidate["oracle_any"] == candidate["answered_rate"] == 1
    assert candidate["compute"]["total_gen"] == {"mean": 20, "p50": 20, "p90": 20}
    lift = candidate["paired_lift"]
    assert lift["n_tasks"] == 4 and lift["difference"] == lift["low"] == lift["high"] == 1
    assert lift["permutation_p"] == 0.125
    assert lift["mcnemar_p"] is None and lift["test"] == "sign_flip"
    table = result.handle.file("markdown").read_text()
    assert "Tasks | Episodes" in table and "95% CI" in table
    assert "1.000 [1.000, 1.000]" in table
    assert "20.0/20.0/20.0" in table and "+1.000 [+1.000, +1.000]" in table
    assert "within-harness" in table and "critical-path tokens" in table
    assert Report.load(result.manifest).n_cells == 2
    assert result.handle.inputs[0].resolve() == source.manifest_path
    before = result.handle.file("results").read_bytes()
    assert (
        await run_verb("eval report", cfg, out=tmp_path / "report")
    ).status is RunStatus.COMPLETE
    assert result.handle.file("results").read_bytes() == before


async def test_different_harness_is_not_paired(tmp_path: Path) -> None:
    source = make_scores(tmp_path / "scores")
    rows = json_rows(source.file("rows"))
    for row in rows:
        if row["protocol"] == "candidate":
            row["harness"] = "different grader or tasks"
    source.file("rows").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = await run_verb(
        "eval report",
        ReportConfig(str(source.manifest_path), "protocol=base", ["protocol"]),
        out=tmp_path / "report",
    )
    candidate = json_rows(result.handle.file("results"))[1]
    assert candidate["paired_lift"] is None
    assert "different taskset or environment" in candidate["comparison_note"]


async def test_report_unknown_group_and_baseline_fail(tmp_path: Path) -> None:
    from marli.errors import ConfigError

    source = make_scores(tmp_path / "scores")
    cfg = ReportConfig(str(source.manifest_path), group_by=["unknown"])
    with pytest.raises(ConfigError, match="group_by"):
        await run_verb("eval report", cfg, out=tmp_path / "badgroup")
    with pytest.raises(ConfigError, match="baseline"):
        await run_verb(
            "eval report",
            replace(cfg, group_by=["protocol"], baseline="absent"),
            out=tmp_path / "badbaseline",
        )


def test_cli_reports_multiple_labelled_scores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    combined = make_scores(tmp_path / "combined")
    sources = []
    for label in ("base", "candidate"):
        root = tmp_path / label
        root.mkdir()
        rows = [row for row in json_rows(combined.file("rows")) if row["protocol"] == label]
        (root / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        scores = Scores(root=root, rows="scores.jsonl", n=len(rows))
        scores.save()
        sources.append(f"{label}={scores.manifest_path}")
    args = [
        "eval",
        "report",
        f"scores={os.pathsep.join(sources)}",
        "baseline=base",
        "--out",
        str(tmp_path / "report"),
    ]
    assert main(args) == 0
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 1
    payload = json.loads(output[0])
    assert payload["n_cells"] == 2 and any("failed" in msg for msg in payload["warnings"])
    candidate = json_rows(tmp_path / "report" / "results.jsonl")[1]
    assert candidate["paired_lift"]["difference"] == 1
