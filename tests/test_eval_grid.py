"""Grid's CLI shares child-run recovery and emits exactly one machine-readable result."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_eval_rollout import json_rows, make_taskset

from marli.errors import ConfigError
from marli.eval import grid as grid_module
from marli.eval.grid import CellSpec, GridConfig, RolloutDefaults
from marli.eval.policies import PolicySpec
from marli.eval.report import Report
from marli.eval.rollout import EpisodeSet
from marli.verbs import run_verb


def test_grid_cli_end_to_end_and_child_resume(tmp_path: Path) -> None:
    source = make_taskset(tmp_path / "tasks", 3)
    cfg = {
        "common": {
            "tasks": str(source.manifest_path),
            "env": "eval_arith",
            "policies": {
                "test": {"ref": "scripted:test_eval_rollout:wrong_policy", "renderer": "fake"}
            },
            "seating": {"solver": "test"},
            "episodes_per_task": 2,
            "limits": {"call": {"max_tokens": 256}},
        },
        "cells": [
            {"label": "base"},
            {
                "label": "candidate",
                "policies": {
                    "test": {
                        "ref": "scripted:test_eval_rollout:correct_policy",
                        "renderer": "fake",
                    }
                },
                "limits": {"agent": {"max_calls": 3}},
            },
        ],
        "baseline": "base",
    }
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(cfg))
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "grid"

    def cli(*extra: str) -> dict[str, Any]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "marli.cli.main",
                "eval",
                "grid",
                str(path),
                *extra,
                "--out",
                str(out),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")]),
            },
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(result.stdout.splitlines()) == 1
        return json.loads(result.stdout)

    result = cli()
    assert result["ok"] and result["kind"] == "report" and result["n_cells"] == 2
    assert (out / "RESULTS.md").is_file()
    report = Report.load(out)
    assert len(report.inputs) == 2
    rows = json_rows(report.file("results"))
    assert rows[1]["paired_lift"]["difference"] == 1
    assert rows[1]["paired_lift"]["n_tasks"] == 3
    assert rows[1]["paired_lift"]["permutation_p"] == 0.25
    for label in ("base", "candidate"):
        episodes = EpisodeSet.load(out / "cells" / label / "rollout")
        assert episodes.n == 6
    child = out / "cells" / "base" / "rollout" / "episodes.jsonl"
    before = child.read_bytes()
    assert cli("parallel_cells=2")["status"] == "complete"
    # Losing the parent manifest simulates interruption after child runs completed.
    report.manifest_path.unlink()
    assert cli()["status"] == "resume"
    assert child.read_bytes() == before
    assert len(json_rows(report.file("results"))) == 2
    saved = yaml.safe_load((out / "cells" / "candidate" / "rollout" / "config.yaml").read_text())
    assert saved["limits"]["call"]["max_tokens"] == 256
    assert saved["limits"]["agent"]["max_calls"] == 3


@pytest.mark.parametrize("label", ["../escape", "..", ".", "a/b", "", "/absolute"])
def test_cell_label_cannot_escape_out(label: str) -> None:
    with pytest.raises(ConfigError, match="directory"):
        CellSpec(label)


def test_grid_validation() -> None:
    with pytest.raises(ConfigError, match="unique"):
        GridConfig(cells=[CellSpec("same"), CellSpec("same")])
    with pytest.raises(ConfigError, match="baseline"):
        GridConfig(cells=[CellSpec("a")], baseline="missing")


@pytest.mark.parametrize("parallel_cells", [1, 2])
async def test_parallel_cells_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parallel_cells: int
) -> None:
    taskset = make_taskset(tmp_path / "tasks", 1)
    common = RolloutDefaults(
        tasks=str(taskset.manifest_path),
        env="eval_arith",
        policies={"test": PolicySpec("scripted:test_eval_rollout:correct_policy", renderer="fake")},
        seating={"solver": "test"},
    )
    cfg = GridConfig(
        cells=[CellSpec(str(i)) for i in range(3)],
        common=common,
        parallel_cells=parallel_cells,
    )
    active = peak = 0

    async def observe(verb: str, cfg: Any, **kwargs: Any) -> Any:
        nonlocal active, peak
        if verb != "eval rollout":
            return await run_verb(verb, cfg, **kwargs)
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)
            return await run_verb(verb, cfg, **kwargs)
        finally:
            active -= 1

    monkeypatch.setattr(grid_module, "run_verb", observe)
    result = await run_verb("eval grid", cfg, out=tmp_path / "grid")
    assert result.handle.n_cells == 3 and peak == parallel_cells and active == 0
