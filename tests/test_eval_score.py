"""Scoring preserves compute and distinguishes system decisions from agent discoveries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from _marli_test_envs import ArithEnv
from test_eval_rollout import json_rows, make_taskset, rollout_config

from marli.eval.policies import PolicySpec
from marli.eval.score import ScoreConfig, Scores
from marli.interact.records import read_episodes
from marli.interact.system import PROTOCOLS, Protocol, RoleSpec, SystemIO
from marli.interact.types import Outcome
from marli.rundir import RunDir, RunStatus
from marli.verbs import run_verb


@dataclass(frozen=True)
class PeersConfig:
    n: int = 2


@PROTOCOLS.register("eval_peers")
class Peers(Protocol):
    name = "eval_peers"
    config_type = PeersConfig

    def __init__(self, config: PeersConfig) -> None:
        self.config = config

    def roles(self) -> list[RoleSpec]:
        return [RoleSpec(role, ("submit",), "Solve.") for role in ("good", "bad")]

    async def run(self, io: SystemIO) -> Outcome:
        handles = [
            await io.start_agent(
                role, agent_id=role, seat_key=(role, 0), first_message=io.env.task_message(role)
            )
            for role in ("good", "bad")
        ]
        good, bad = await io.wait(handles)
        return Outcome(bad.submission, {"good": good.submission, "bad": bad.submission}, "test")


async def test_multi_agent_score_compute_oracle_and_regrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = rollout_config(make_taskset(tmp_path / "tasks", 2), protocol="eval_peers")
    cfg.policies["wrong"] = PolicySpec("scripted:test_eval_rollout:wrong_policy", renderer="fake")
    cfg.seating = {"good": "test", "bad": "wrong"}
    result = await run_verb("eval rollout", cfg, out=tmp_path / "rollout")
    scored = await run_verb("eval score", ScoreConfig(str(result.manifest)), out=tmp_path / "score")
    rows = json_rows(scored.handle.file("rows"))
    assert len(rows) == 2 and scored.handle.n == 2
    assert Scores.load(scored.manifest).inputs[0].resolve() == result.manifest
    original = {
        episode.episode_id: episode
        for episode, _ in read_episodes(result.handle.root, with_tokens=False)
    }
    for row in rows:
        assert row["correct"] == 0 and row["oracle_any"] is True
        assert row["own_correct"] == {"good": 1, "bad": 0}
        assert row["answered"] and row["ok"]
        assert row["n_agents"] == 2 and row["calls"] == 2 and row["cp_calls"] == 1
        assert row["protocol"] == "eval_peers"
        for metric, value in original[row["episode_id"]].metrics.items():
            assert row[metric] == value
    before = result.handle.file("episodes").read_bytes()
    setup = teardown = 0

    async def new_grade(self: ArithEnv, submission: str | None) -> dict[str, float]:
        return {"correct": float(submission == "4")}

    async def count_setup(self: ArithEnv) -> None:
        nonlocal setup
        setup += 1

    async def count_teardown(self: ArithEnv) -> None:
        nonlocal teardown
        teardown += 1

    monkeypatch.setattr(ArithEnv, "grade", new_grade)
    monkeypatch.setattr(ArithEnv, "setup", count_setup)
    monkeypatch.setattr(ArithEnv, "teardown", count_teardown)
    regraded = await run_verb(
        "eval score", ScoreConfig(str(result.manifest), regrade=True), out=tmp_path / "regrade"
    )
    new_rows = json_rows(regraded.handle.file("rows"))
    assert all(
        row["correct"] == 1 and row["own_correct"] == {"bad": 1, "good": 0} for row in new_rows
    )
    assert setup == teardown == 2
    assert result.handle.file("episodes").read_bytes() == before


async def test_score_resume_and_equivalent_answer_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = rollout_config(make_taskset(tmp_path / "tasks", 1), episodes_per_task=3)
    cfg.policies["test"] = PolicySpec("scripted:test_eval_rollout:mixed_policy", renderer="fake")
    episodes = await run_verb("eval rollout", cfg, out=tmp_path / "rollout")
    original = RunDir.append_row
    out = tmp_path / "score"

    def crash(run: RunDir, rel: str, row: Any) -> None:
        if rel == "scores.jsonl" and row["episode_idx"] == 1:
            raise RuntimeError("score crash")
        original(run, rel, row)

    score_cfg = ScoreConfig(str(episodes.manifest))
    with monkeypatch.context() as patch:
        patch.setattr(RunDir, "append_row", crash)
        with pytest.raises(RuntimeError, match="score crash"):
            await run_verb("eval score", score_cfg, out=out)
    assert len(json_rows(out / "scores.jsonl")) == 1
    result = await run_verb("eval score", score_cfg, out=out)
    assert result.status is RunStatus.RESUME
    rows = json_rows(result.handle.file("rows"))
    assert len({row["episode_id"] for row in rows}) == 3
    assert [row["answer_group"] for row in rows] == [0, 0, 1]
    assert [row["correct"] for row in rows] == [1, 1, 0]


async def test_unanswered_and_failed_rows_are_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marli.eval import rollout as rollout_module

    original = rollout_module.run_episode

    async def unanswered(spec: Any) -> Any:
        episode, tokens = await original(spec)
        return replace(
            episode,
            outcome=Outcome(None, {"solver0": None}, "single"),
            grades={"_system": {"correct": 0.0}},
            ok=False,
        ), tokens

    monkeypatch.setattr(rollout_module, "run_episode", unanswered)
    result = await run_verb(
        "eval rollout",
        rollout_config(make_taskset(tmp_path / "tasks", 1)),
        out=tmp_path / "rollout",
    )
    scored = await run_verb("eval score", ScoreConfig(str(result.manifest)), out=tmp_path / "score")
    (row,) = json_rows(scored.handle.file("rows"))
    assert row["correct"] == 0 and not row["answered"] and not row["ok"]
    assert row["oracle_any"] is False and row["own_correct"] == {"solver0": 0}
