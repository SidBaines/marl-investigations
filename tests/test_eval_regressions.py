"""Reviewer probes exercise real policy failures, saved identity, and task-level inference."""

from __future__ import annotations

import json
import os
import random
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_eval_rollout import json_rows, make_taskset, rollout_config

from marli import verbs
from marli.budget import SpendGuard
from marli.cli.main import main
from marli.config import from_mappings, save
from marli.errors import BackendError, BudgetExceededError, HashMismatchError
from marli.eval import rollout as rollout_module
from marli.eval.grid import CellSpec, GridConfig, RolloutDefaults
from marli.eval.grid import rollout_config as cell_config
from marli.eval.policies import PolicySpec
from marli.eval.report import ReportConfig
from marli.eval.score import ScoreConfig, Scores
from marli.eval.stats import bootstrap_mean, paired_comparison, sign_flip_p, wilson_ci
from marli.eval.store import read_episodes
from marli.policy.base import CallMeta, Sample, SamplingSpec
from marli.policy.scripted import ScriptedPolicy, Turn, from_callable
from marli.render.fake import FakeRenderer
from marli.rundir import RunDir, RunStatus
from marli.verbs import run_verb


class ChargingPolicy(ScriptedPolicy):
    def __init__(self, guard: SpendGuard, on_sample: Callable[[CallMeta | None], None]) -> None:
        renderer = FakeRenderer()
        script = from_callable(
            lambda ctx: Turn(tool_calls=(("submit", {"answer": "5"}),)), renderer
        )
        super().__init__("charging", renderer, script)
        self.guard = guard
        self.on_sample = on_sample

    async def sample(
        self,
        prompt_ids: Sequence[int],
        spec: SamplingSpec,
        *,
        seed: int,
        meta: CallMeta | None = None,
    ) -> Sample:
        self.on_sample(meta)
        self.guard.charge(0.4, self.policy_id)
        sample = await super().sample(prompt_ids, spec, seed=seed, meta=meta)
        return replace(sample, usage=replace(sample.usage, cost_usd=0.4))


def install_charging(
    monkeypatch: pytest.MonkeyPatch,
    on_sample: Callable[[CallMeta | None], None],
) -> None:
    async def build(specs: dict[str, PolicySpec], *, spend: SpendGuard) -> Any:
        return {key: ChargingPolicy(spend, on_sample) for key in specs}, {
            key: FakeRenderer for key in specs
        }

    monkeypatch.setattr(rollout_module, "build_policies", build)


def test_policy_internal_budget_exit_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_charging(monkeypatch, lambda meta: None)
    config = rollout_config(make_taskset(tmp_path / "tasks", 6), concurrency=2, max_usd=1)
    path = save(config, tmp_path / "rollout.yaml")
    out = tmp_path / "rollout"
    args = ["eval", "rollout", str(path), "--out", str(out)]
    assert main(args) == 4
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 1 and json.loads(output[0])["exit_code"] == 4
    rows = json_rows(out / "episodes.jsonl")
    assert len(rows) < 6 and all(row["ok"] for row in rows)
    assert not (out / "episodes.json").exists()
    assert main([*args, "max_usd=10"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n"] == 6 and payload["n_failed"] == 0
    assert all(row["ok"] for row in json_rows(out / "episodes.jsonl"))


async def test_legacy_budget_failure_is_never_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = rollout_module.run_episode

    async def failed(spec: Any) -> Any:
        episode, tokens = await original(spec)
        return replace(episode, ok=False, errors=("BudgetExceededError: charged",)), tokens

    monkeypatch.setattr(rollout_module, "run_episode", failed)
    out = tmp_path / "rollout"
    with pytest.raises(BudgetExceededError):
        await run_verb("eval rollout", rollout_config(make_taskset(tmp_path / "tasks", 1)), out=out)
    assert json_rows(out / "episodes.jsonl") == []


async def test_retry_failed_completed_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = True
    calls: list[str] = []

    def sample(meta: CallMeta | None) -> None:
        assert meta is not None
        calls.append(meta.episode_id)
        if failed and meta.episode_id.startswith("t1/"):
            raise BackendError("transient 503")

    install_charging(monkeypatch, sample)
    monkeypatch.setattr(warnings, "showwarning", verbs._showwarning)
    cfg = rollout_config(make_taskset(tmp_path / "tasks", 3), record_tokens=True)
    out = tmp_path / "rollout"
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        first = await run_verb("eval rollout", cfg, out=out)
    assert first.handle.n_failed == 1 and any("failed" in msg for msg in first.warnings)
    frozen = await run_verb("eval rollout", replace(cfg, retry_failed=False), out=out)
    assert frozen.status is RunStatus.COMPLETE and frozen.handle.n_failed == 1
    scores = await run_verb("eval score", ScoreConfig(str(first.manifest)), out=tmp_path / "score")
    report = await run_verb(
        "eval report", ReportConfig(str(scores.manifest)), out=tmp_path / "report"
    )
    row = json_rows(report.handle.file("results"))[0]
    assert row["accuracy"] == pytest.approx(2 / 3) and row["accuracy_ok"] == 1
    assert row["n_failed"] == 1 and any("failed" in msg for msg in report.warnings)
    saved = {row["episode_id"]: row for row in json_rows(out / "episodes.jsonl")}
    failed = False
    calls.clear()
    second = await run_verb("eval rollout", cfg, out=out)
    assert second.handle.n_failed == 0
    assert len(calls) == 1 and calls[0].startswith("t1/")
    rows = json_rows(out / "episodes.jsonl")
    assert len(rows) == len(json_rows(out / "tokens.jsonl")) == 3
    for row in rows:
        if not row["episode_id"].startswith("t1/"):
            assert row == saved[row["episode_id"]]


async def test_reader_last_attempt_and_rollout_compaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marli.interact import records

    cfg = rollout_config(make_taskset(tmp_path / "tasks", 1), record_tokens=True)
    out = tmp_path / "rollout"
    first = await run_verb("eval rollout", cfg, out=out)
    first.manifest.unlink()
    episode, tokens = next(records.read_episodes(out, with_tokens=True))
    with RunDir(
        out, kind="eval rollout", manifest_name="episodes.json", config_hash=first.config_hash
    ) as run:
        records.write_episode(
            run.append_row,
            replace(episode, ok=False, errors=("transient",)),
            tokens,
            record_tokens=True,
        )
    assert not list(read_episodes(out))[0].ok
    resumed = await run_verb("eval rollout", cfg, out=out)
    assert resumed.handle.n == 1 and resumed.handle.n_failed == 0
    assert len(json_rows(out / "episodes.jsonl")) == len(json_rows(out / "tokens.jsonl")) == 1
    assert next(records.read_episodes(out, with_tokens=True))[0].ok


async def test_effective_limits_and_policy_defaults(tmp_path: Path) -> None:
    cfg = rollout_config(make_taskset(tmp_path / "tasks", 1), protocol="multi_session_s3_notes")
    cfg.policies["test"].renderer = None
    result = await run_verb("eval rollout", cfg, out=tmp_path / "rollout")
    assert cfg.limits.session.max_sessions == 1
    assert result.handle.meta["limits"]["session"]["max_sessions"] == 3
    # multi_session splits the agent budget across its sessions (protocol-owned)
    agent = cfg.limits.agent
    assert result.handle.meta["limits"]["session"]["max_gen_tokens"] == (
        (agent.max_gen_tokens - agent.final_reserve) // 3
    )
    assert result.handle.meta["policy_specs"]["test"]["renderer"] == "fake"


async def test_renderer_disagreement_warns_and_records_resolved_model(tmp_path: Path) -> None:
    cfg = rollout_config(make_taskset(tmp_path / "tasks", 1))
    cfg.policies["test"].model = "qwen3_8b"
    with pytest.warns(UserWarning, match="disagrees with model"):
        result = await run_verb("eval rollout", cfg, out=tmp_path / "rollout")
    spec = result.handle.meta["policy_specs"]["test"]
    assert spec["renderer"] == "fake" and spec["model"] == "qwen3_8b"


def test_sparse_policy_override_preserves_common_fields() -> None:
    cfg = from_mappings(
        GridConfig,
        {
            "common": {
                "policies": {
                    "p": {
                        "ref": "scripted:test_eval_rollout:correct_policy",
                        "renderer": "fake",
                        "sampling": {"temperature": 0.3, "top_p": 0.7},
                    }
                }
            },
            "cells": [{"label": "a", "policies": {"p": {"sampling": {"top_p": 1.0}}}}],
        },
    )
    resolved = cell_config(cfg.common, cfg.cells[0]).policies["p"]
    assert resolved.renderer == "fake"
    assert resolved.sampling.temperature == 0.3 and resolved.sampling.top_p == 1


@pytest.mark.parametrize("parallel", [1, 2])
async def test_grid_spend_includes_completed_and_incomplete_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parallel: int,
) -> None:
    install_charging(monkeypatch, lambda meta: None)
    common = RolloutDefaults(
        **vars(rollout_config(make_taskset(tmp_path / "tasks", 2), concurrency=1))
    )
    cfg = GridConfig(
        cells=[CellSpec("a"), CellSpec("b"), CellSpec("c")],
        common=common,
        parallel_cells=parallel,
        max_usd=1.0,
    )
    out = tmp_path / "grid"
    with pytest.raises(BudgetExceededError):
        await run_verb("eval grid", cfg, out=out)
    assert not (out / "report.json").exists()
    progress = [json.loads(path.read_text()) for path in out.glob("cells/*/rollout/progress.json")]
    assert sum(item["spend"]["spent_usd"] for item in progress) > 1.0
    completed = {path: path.read_bytes() for path in out.glob("cells/*/rollout/episodes.json")}
    result = await run_verb("eval grid", replace(cfg, max_usd=5), out=out)
    assert result.handle.n_cells == 3
    assert all(path.read_bytes() == before for path, before in completed.items())
    assert all(row["n_failed"] == 0 for row in json_rows(out / "results.jsonl"))


def test_task_sign_flips_include_even_repeat_ties() -> None:
    cell = {str(i): [1] * 8 if i < 5 else [0] * 8 for i in range(10)}
    baseline = {str(i): [1, 0] * 4 for i in range(10)}
    result = paired_comparison(cell, baseline)
    assert result.difference == 0 and result.permutation_p == 1 and result.mcnemar_p is None
    assert result.wins == result.losses == 5
    assert sign_flip_p([0.5] * 4) == 0.125
    assert sign_flip_p([0] * 25) == 1
    state = random.getstate()
    assert sign_flip_p([0.5] * 21, seed=4) == sign_flip_p([0.5] * 21, seed=4)
    assert random.getstate() == state


async def test_multi_scores_input_hash_ci_and_compute_report(tmp_path: Path) -> None:
    sources = {}
    for label in ("base", "swarm"):
        root = tmp_path / label
        root.mkdir()
        rows = [
            {
                "task_id": f"t{task}",
                "episode_id": f"{label}/{task}/{repeat}",
                "episode_idx": repeat,
                "protocol": label,
                "policy": "fake",
                "taskset": "tasks",
                "harness": "same",
                "correct": float(task < 5),
                "answered": True,
                "oracle_any": task < 5,
                "ok": True,
                "total_gen": 10 if label == "base" else 20,
                "cp_tokens": 10,
                "total_uncached": 30,
            }
            for task in range(10)
            for repeat in range(8)
        ]
        (root / "scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        source = Scores(root=root, rows="scores.jsonl", n=80, meta={"cost_usd": 1.25})
        source.save()
        sources[label] = source
    cfg = ReportConfig(
        os.pathsep.join(f"{k}={v.root}" for k, v in sources.items()), baseline="base"
    )
    result = await run_verb("eval report", cfg, out=tmp_path / "report")
    estimate = bootstrap_mean([1] * 5 + [0] * 5)
    rows = json_rows(result.handle.file("results"))
    assert rows[0]["accuracy_ci"] == [estimate.low, estimate.high]
    assert rows[0]["accuracy_ci"] != list(wilson_ci(40, 80))
    assert rows[0]["cost_usd"] == 1.25
    markdown = result.handle.file("markdown").read_text()
    assert "Accuracy vs compute" in markdown and "task_bootstrap" in markdown
    assert "Total uncached" in markdown and "Not compute-matched" in markdown
    assert "swarm" in markdown
    sources["swarm"].manifest_path.write_text(sources["swarm"].manifest_path.read_text() + "\n")
    with pytest.raises(HashMismatchError):
        await run_verb("eval report", cfg, out=tmp_path / "report")


async def test_regrade_is_part_of_pairing_harness(tmp_path: Path) -> None:
    episodes = await run_verb(
        "eval rollout",
        rollout_config(make_taskset(tmp_path / "tasks", 1)),
        out=tmp_path / "rollout",
    )
    raw = await run_verb("eval score", ScoreConfig(str(episodes.manifest)), out=tmp_path / "raw")
    regraded = await run_verb(
        "eval score", ScoreConfig(str(episodes.manifest), regrade=True), out=tmp_path / "regraded"
    )
    assert raw.handle.meta["regrade"] is False and regraded.handle.meta["regrade"] is True
    report = await run_verb(
        "eval report",
        ReportConfig(f"raw={raw.manifest}{os.pathsep}regraded={regraded.manifest}", baseline="raw"),
        out=tmp_path / "report",
    )
    row = json_rows(report.handle.file("results"))[1]
    assert row["paired_lift"] is None and "regrade" in row["comparison_note"]


async def test_completed_grid_adds_only_new_cell(tmp_path: Path) -> None:
    common = RolloutDefaults(**vars(rollout_config(make_taskset(tmp_path / "tasks", 1))))
    cfg = GridConfig(cells=[CellSpec("base")], common=common, baseline="base")
    out = tmp_path / "grid"
    await run_verb("eval grid", cfg, out=out)
    saved = out / "cells" / "base" / "rollout" / "episodes.jsonl"
    stat, content = saved.stat(), saved.read_bytes()
    added = replace(cfg, cells=[CellSpec("base"), CellSpec("extra")])
    result = await run_verb("eval grid", added, out=out)
    assert result.handle.n_cells == 2
    assert (saved.stat().st_ino, saved.stat().st_mtime_ns) == (stat.st_ino, stat.st_mtime_ns)
    assert saved.read_bytes() == content


async def test_completed_grid_changed_cell_and_taskset_fail_on_child(tmp_path: Path) -> None:
    taskset = make_taskset(tmp_path / "tasks", 1)
    common = RolloutDefaults(**vars(rollout_config(taskset)))
    cfg = GridConfig(cells=[CellSpec("base")], common=common)
    out = tmp_path / "grid"
    await run_verb("eval grid", cfg, out=out)
    with pytest.raises(HashMismatchError, match="cells/base/rollout"):
        await run_verb(
            "eval grid", replace(cfg, cells=[CellSpec("base", episodes_per_task=2)]), out=out
        )
    make_taskset(taskset.root, 2)
    with pytest.raises(HashMismatchError, match="cells/base/rollout"):
        await run_verb("eval grid", cfg, out=out)


async def test_existing_scores_cell_and_external_cell_storage(tmp_path: Path) -> None:
    common = RolloutDefaults(**vars(rollout_config(make_taskset(tmp_path / "tasks", 1))))
    store = tmp_path / "shared-cells"
    cfg = GridConfig(cells=[CellSpec("base")], common=common, cells_dir=str(store))
    await run_verb("eval grid", cfg, out=tmp_path / "grid1")
    rows = store / "base" / "rollout" / "episodes.jsonl"
    before = rows.stat().st_mtime_ns
    await run_verb("eval grid", cfg, out=tmp_path / "grid2")
    assert rows.stat().st_mtime_ns == before
    loaded = GridConfig(cells=[CellSpec("loaded", scores=str(store / "base" / "score"))])
    result = await run_verb("eval grid", loaded, out=tmp_path / "grid3")
    assert result.handle.n_cells == 1
    assert not (tmp_path / "grid3" / "cells").exists()


@pytest.mark.parametrize("cli_force", [False, True])
async def test_grid_force_replaces_only_changed_cell(tmp_path: Path, cli_force: bool) -> None:
    common = RolloutDefaults(**vars(rollout_config(make_taskset(tmp_path / "tasks", 1))))
    cfg = GridConfig(cells=[CellSpec("base"), CellSpec("candidate")], common=common)
    out = tmp_path / "grid"
    await run_verb("eval grid", cfg, out=out)
    path = out / "cells" / "base" / "rollout" / "episodes.jsonl"
    before = path.stat().st_mtime_ns, path.read_bytes()
    cfg = replace(
        cfg,
        cells=[CellSpec("base"), CellSpec("candidate", episodes_per_task=2)],
        force_cells=[] if cli_force else ["candidate"],
    )
    result = await run_verb("eval grid", cfg, out=out, force=cli_force)
    assert result.status is RunStatus.RESUME
    assert (path.stat().st_mtime_ns, path.read_bytes()) == before
    rows = json_rows(result.handle.file("results"))
    assert [row["n_episodes"] for row in rows] == [1, 2]


async def test_parallel_grid_allocates_only_to_unfinished_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_charging(monkeypatch, lambda meta: None)
    common = RolloutDefaults(**vars(rollout_config(make_taskset(tmp_path / "tasks", 2))))
    cfg = GridConfig(cells=[CellSpec("a")], common=common, parallel_cells=2, max_usd=1.6)
    out = tmp_path / "grid"
    await run_verb("eval grid", cfg, out=out)
    result = await run_verb(
        "eval grid", replace(cfg, cells=[CellSpec("a"), CellSpec("b")]), out=out
    )
    assert result.handle.n_cells == 2
    assert json.loads((out / "progress.json").read_text())["cost_usd"] == pytest.approx(1.6)


async def test_grid_retries_failures_and_refreshes_interrupted_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed = True

    def sample(meta: CallMeta | None) -> None:
        if failed and meta is not None and meta.episode_id.startswith("t1/"):
            raise BackendError("temporary unavailable")

    install_charging(monkeypatch, sample)
    common = RolloutDefaults(**vars(rollout_config(make_taskset(tmp_path / "tasks", 2))))
    cfg = GridConfig(cells=[CellSpec("a")], common=common)
    out = tmp_path / "grid"
    first = await run_verb("eval grid", cfg, out=out)
    assert any("rollout has 1 failed" in message for message in first.warnings)
    failed = False
    from marli.eval import grid as grid_module

    async def crash(verb: str, *args: Any, **kwargs: Any) -> Any:
        if verb == "eval score":
            raise RuntimeError("interrupted before refreshed scoring")
        return await run_verb(verb, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(grid_module, "run_verb", crash)
        with pytest.raises(RuntimeError, match="interrupted before"):
            await run_verb("eval grid", cfg, out=out)
    result = await run_verb("eval grid", cfg, out=out)
    row = json_rows(result.handle.file("results"))[0]
    assert row["accuracy"] == 1 and row["n_failed"] == 0
    assert not result.warnings


async def test_grid_regrade_requires_cell_force_and_records_setting(tmp_path: Path) -> None:
    common = RolloutDefaults(**vars(rollout_config(make_taskset(tmp_path / "tasks", 1))))
    cfg = GridConfig(cells=[CellSpec("a")], common=common)
    out = tmp_path / "grid"
    await run_verb("eval grid", cfg, out=out)
    changed = replace(cfg, cells=[CellSpec("a", regrade=True)])
    with pytest.raises(HashMismatchError, match="cells/a/score"):
        await run_verb("eval grid", changed, out=out)
    await run_verb("eval grid", changed, out=out, force=True)
    assert Scores.load(out / "cells" / "a" / "score").meta["regrade"] is True


async def test_grid_preserves_forced_run_spend_after_interrupted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_charging(monkeypatch, lambda meta: None)
    common = RolloutDefaults(**vars(rollout_config(make_taskset(tmp_path / "tasks", 1))))
    cfg = GridConfig(cells=[CellSpec("a")], common=common, max_usd=1.3)
    out = tmp_path / "grid"
    await run_verb("eval grid", cfg, out=out)
    changed = replace(cfg, cells=[CellSpec("a", episodes_per_task=2)], force_cells=["a"])
    original = RunDir.write_progress

    def crash(run: RunDir, data: Mapping[str, Any]) -> None:
        if run.out == out and not data["forced_attempts"]:
            raise OSError("crash before grid spend checkpoint")
        original(run, data)

    with monkeypatch.context() as patch:
        patch.setattr(RunDir, "write_progress", crash)
        with pytest.raises(OSError, match="grid spend checkpoint"):
            await run_verb("eval grid", changed, out=out)
    with pytest.raises(BudgetExceededError):
        await run_verb("eval grid", replace(changed, force_cells=[], max_usd=1.0), out=out)
    result = await run_verb("eval grid", replace(changed, force_cells=[]), out=out)
    assert result.handle.n_cells == 1
    assert json.loads((out / "progress.json").read_text())["cost_usd"] == pytest.approx(1.2)
