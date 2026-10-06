"""Saved synthetic episodes pin difficulty selection independently of rollout backends."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from marli.cli.main import main
from marli.config import compose
from marli.data.filter import FilterConfig
from marli.envs.base import Task
from marli.errors import ConfigError, HashMismatchError
from marli.handles import InputRef
from marli.interact import records
from marli.interact.types import Episode, Outcome
from marli.rundir import RunDir, RunStatus
from marli.tasks.taskset import TaskSet, read_tasks, write_tasks
from marli.verbs import run_verb


@pytest.fixture
def taskset(tmp_path: Path) -> TaskSet:
    tasks = [
        Task(name, f"Synthetic {name}", "answer", {"topic": "fixture"})
        for name in ("mixed", "zero", "one", "few", "absent", "fractional")
    ]
    root = tmp_path / "tasks"
    write_tasks(root, tasks)
    handle = TaskSet(
        root=root,
        tasks="tasks.jsonl",
        source="synthetic",
        split="train",
        kind="math",
        n=len(tasks),
        answer_format="latex",
        commit_text=False,
        meta={"origin": "synthetic fixture"},
    )
    handle.save()
    return handle


def episode(task_id: str, index: int, value: float, *, ok: bool = True) -> Episode:
    return Episode(
        episode_id=f"{task_id}/e{index}",
        group_id=task_id,
        episode_idx=index,
        task_id=task_id,
        protocol="single",
        config_hash="synthetic",
        backend="scripted",
        agents=(),
        segments=(),
        calls=(),
        events=(),
        workspace_log=(),
        outcome=Outcome("synthetic", {}, "single"),
        grades={"_system": {"correct": value, "other": 1.0}, "peer0": {"correct": 1.0}},
        limits_hit={},
        metrics={},
        replayable=True,
        ok=ok,
    )


def save_episodes(root: Path, episodes: list[Episode], taskset: TaskSet) -> Path:
    with RunDir(root, kind="fixture", manifest_name="episodes.json", config_hash="fixture") as run:
        for item in episodes:
            records.write_episode(run.append_row, item, {}, record_tokens=False)
    # Filtering needs recorded TaskSet provenance even for hand-built episodes.
    (root / "episodes.json").write_text(
        json.dumps(
            {
                "kind": "episodes",
                "manifest_version": 1,
                "inputs": [
                    {
                        "kind": "taskset",
                        "path": str(taskset.manifest_path),
                        "sha256": taskset.sha256(),
                    }
                ],
            }
        )
        + "\n"
    )
    return root


@pytest.fixture
def rollouts(tmp_path: Path, taskset: TaskSet) -> Path:
    episodes = []
    for name, values in {
        "mixed": [0.0, 1.0, 1.0],
        "zero": [0.0, 0.0],
        "one": [1.0, 1.0],
        "few": [1.0],
        "fractional": [0.2, 0.6],
    }.items():
        episodes.extend(episode(name, i, value) for i, value in enumerate(values))
    episodes.extend(
        [
            episode("mixed", 3, 0.0, ok=False),
            replace(episode("few", 1, 0.0, ok=False), grades={}),
            episode("absent", 0, 1.0, ok=False),
        ]
    )
    return save_episodes(tmp_path / "episodes", episodes, taskset)


async def test_pass_rates_metadata_and_counts(
    tmp_path: Path, taskset: TaskSet, rollouts: Path
) -> None:
    original = taskset.file("tasks").read_bytes()
    result = await run_verb(
        "data filter",
        FilterConfig(tasks=str(taskset.root), episodes=str(rollouts)),
        out=tmp_path / "out",
    )
    handle = TaskSet.load(result.manifest)
    tasks = read_tasks(handle)
    assert [task.task_id for task in tasks] == ["mixed", "fractional"]
    assert [task.meta["pass_rate"] for task in tasks] == pytest.approx([2 / 3, 0.4])
    assert all(task.meta["topic"] == "fixture" for task in tasks)
    assert all(task.answer == "answer" for task in tasks)
    assert [task.prompt for task in tasks] == ["Synthetic mixed", "Synthetic fractional"]
    assert handle.summary() == {"n": 2, "source": "synthetic"}
    assert handle.kind == taskset.kind
    assert handle.split == taskset.split
    assert handle.answer_format == taskset.answer_format
    assert handle.commit_text is False
    assert handle.meta["origin"] == "synthetic fixture"
    assert handle.meta["filter"] == {
        "lo": 0.0,
        "hi": 1.0,
        "inclusive": False,
        "min_episodes": 2,
        "metric": "correct",
        "counts": {
            "input": 6,
            "kept": 2,
            "dropped_min_episodes": 2,
            "dropped_lo": 1,
            "dropped_hi": 1,
        },
        "ignored_non_ok_episodes": 3,
    }
    assert handle.inputs == (
        InputRef.of(taskset),
        InputRef.from_manifest(rollouts / "episodes.json"),
    )
    assert taskset.file("tasks").read_bytes() == original
    assert not (rollouts / "tokens.jsonl").exists()


@pytest.mark.parametrize(
    "lo,hi,inclusive,expected",
    [
        (0.0, 1.0, True, ["mixed", "zero", "one", "fractional"]),
        (0.4, 2 / 3, False, []),
        (0.4, 2 / 3, True, ["mixed", "fractional"]),
        (0.4, 0.4, True, ["fractional"]),
        (0.4, 0.4, False, []),
    ],
)
async def test_strict_and_inclusive_bounds(
    tmp_path: Path,
    taskset: TaskSet,
    rollouts: Path,
    lo: float,
    hi: float,
    inclusive: bool,
    expected: list[str],
) -> None:
    result = await run_verb(
        "data filter",
        FilterConfig(
            tasks=str(taskset.manifest_path),
            episodes=str(rollouts),
            lo=lo,
            hi=hi,
            inclusive=inclusive,
        ),
        out=tmp_path / "out",
    )
    assert [task.task_id for task in read_tasks(result.handle)] == expected
    counts = result.handle.meta["filter"]["counts"]
    assert counts["input"] == sum(value for key, value in counts.items() if key != "input")


async def test_minimum_episodes_and_custom_metric(
    tmp_path: Path, taskset: TaskSet, rollouts: Path
) -> None:
    cfg = FilterConfig(tasks=str(taskset.root), episodes=str(rollouts), min_episodes=3)
    result = await run_verb("data filter", cfg, out=tmp_path / "min-three")
    assert [task.task_id for task in read_tasks(result.handle)] == ["mixed"]
    assert result.handle.meta["filter"]["counts"]["dropped_min_episodes"] == 5
    custom = await run_verb(
        "data filter",
        replace(cfg, metric="other", min_episodes=1, inclusive=True),
        out=tmp_path / "custom",
    )
    assert [task.task_id for task in read_tasks(custom.handle)] == [
        "mixed",
        "zero",
        "one",
        "few",
        "fractional",
    ]
    assert all(task.meta["pass_rate"] == 1.0 for task in read_tasks(custom.handle))
    assert custom.handle.meta["filter"]["metric"] == "other"


def test_filter_cli_idempotency(
    tmp_path: Path,
    taskset: TaskSet,
    rollouts: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "out"
    args = ["data", "filter", f"tasks={taskset.root}", f"episodes={rollouts}", "--out", str(out)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["kind"] == "taskset"
    assert payload["n"] == 2
    assert payload["source"] == "synthetic"
    assert payload["status"] == "fresh"
    assert payload["manifest"] == str(out / "taskset.json")
    saved = yaml.safe_load((out / "config.yaml").read_text())
    assert saved["tasks"] == str(taskset.manifest_path)
    assert saved["episodes"] == str(rollouts / "episodes.json")
    before = {name: (out / name).read_bytes() for name in ("tasks.jsonl", "taskset.json")}

    def unexpected_read(*args: Any, **kwargs: Any) -> None:
        pytest.fail("completed runs must not read episode rows")

    monkeypatch.setattr(records, "read_episodes", unexpected_read)
    assert main(args) == 0
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out) == {**payload, "status": "complete"}
    assert before == {name: (out / name).read_bytes() for name in before}


async def test_resume_and_input_manifest_hash(
    tmp_path: Path, taskset: TaskSet, rollouts: Path
) -> None:
    cfg = FilterConfig(tasks=str(taskset.root), episodes=str(rollouts))
    out = tmp_path / "out"
    result = await run_verb("data filter", cfg, out=out)
    before = result.handle.file("tasks").read_bytes()
    result.manifest.unlink()
    resumed = await run_verb("data filter", cfg, out=out)
    assert resumed.status is RunStatus.RESUME
    assert resumed.handle.file("tasks").read_bytes() == before
    manifest = rollouts / "episodes.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(HashMismatchError):
        await run_verb("data filter", cfg, out=out)


@pytest.mark.parametrize(
    "grades",
    [{}, {"_system": {}}, {"_system": {"correct": float("nan")}}, {"_system": {"correct": "bad"}}],
)
async def test_invalid_ok_episode_grades_fail_loudly(
    tmp_path: Path, taskset: TaskSet, grades: dict[str, Any]
) -> None:
    root = save_episodes(
        tmp_path / "bad-episodes", [replace(episode("mixed", 0, 1.0), grades=grades)], taskset
    )
    with pytest.raises(ConfigError, match="episode 'mixed/e0'"):
        await run_verb(
            "data filter",
            FilterConfig(tasks=str(taskset.root), episodes=str(root)),
            out=tmp_path / "out",
        )
    assert not (tmp_path / "out" / "taskset.json").exists()


async def test_episodes_from_unknown_task_fail_loudly(tmp_path: Path, taskset: TaskSet) -> None:
    root = save_episodes(tmp_path / "bad-episodes", [episode("unrelated", 0, 1.0)], taskset)
    with pytest.raises(ConfigError, match="unknown task"):
        await run_verb(
            "data filter",
            FilterConfig(tasks=str(taskset.root), episodes=str(root)),
            out=tmp_path / "out",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        ["tasks=tasks"],
        ["episodes=episodes"],
        ["tasks=tasks", "episodes=episodes", "min_episodes=0"],
        ["tasks=tasks", "episodes=episodes", "lo=0.9", "hi=0.1"],
        ["tasks=tasks", "episodes=episodes", "hi=nan"],
        ["tasks=tasks", "episodes=episodes", "metric="],
    ],
)
def test_invalid_config(overrides: list[str]) -> None:
    with pytest.raises(ConfigError):
        compose(FilterConfig, overrides=overrides)


async def test_filter_rejects_recorded_taskset_digest_mismatch(
    tmp_path: Path,
    taskset: TaskSet,
    rollouts: Path,
) -> None:
    changed = replace(taskset, meta={**taskset.meta, "replacement": True})
    changed.save()
    with pytest.raises(ConfigError, match="TaskSet digest"):
        await run_verb(
            "data filter",
            FilterConfig(tasks=str(taskset.root), episodes=str(rollouts)),
            out=tmp_path / "out",
        )


async def test_wall_clock_failures_count_as_zero_and_last_attempt_wins(
    tmp_path: Path,
    taskset: TaskSet,
) -> None:
    root = save_episodes(
        tmp_path / "episodes",
        [
            episode("mixed", 0, 0),
            episode("mixed", 0, 1),
            replace(
                episode("mixed", 1, 1, ok=False),
                grades={},
                limits_hit={"_episode": ("episode.max_wall_s",)},
            ),
        ],
        taskset,
    )
    result = await run_verb(
        "data filter",
        FilterConfig(tasks=str(taskset.root), episodes=str(root)),
        out=tmp_path / "out",
    )
    (task,) = read_tasks(result.handle)
    assert task.task_id == "mixed" and task.meta["pass_rate"] == 0.5
    assert result.handle.meta["filter"]["ignored_non_ok_episodes"] == 0


async def test_paused_rollout_is_rejected(tmp_path: Path, taskset: TaskSet, rollouts: Path) -> None:
    manifest = rollouts / "episodes.json"
    recorded = json.loads(manifest.read_text())
    recorded["meta"] = {"paused": True, "n_tasks_sampled": 2, "n_tasks": 6}
    manifest.write_text(json.dumps(recorded) + "\n")
    with pytest.raises(ConfigError, match="paused rollout \\(2 of 6 tasks"):
        await run_verb(
            "data filter",
            FilterConfig(tasks=str(taskset.root), episodes=str(rollouts)),
            out=tmp_path / "out",
        )


async def test_paused_rollout_with_allow_paused_filters_only_the_sampled_tasks(
    tmp_path: Path, taskset: TaskSet, rollouts: Path
) -> None:
    # The fixture's first three tasks (mixed, zero, one) were "sampled"; few, absent and fractional
    # were not, so fractional (in range when complete) is dropped as unsampled.
    manifest = rollouts / "episodes.json"
    recorded = json.loads(manifest.read_text())
    recorded["meta"] = {"paused": True, "n_tasks_sampled": 3, "n_tasks": 6}
    manifest.write_text(json.dumps(recorded) + "\n")
    result = await run_verb(
        "data filter",
        FilterConfig(tasks=str(taskset.root), episodes=str(rollouts), allow_paused=True),
        out=tmp_path / "out",
    )
    handle = TaskSet.load(result.manifest)
    assert [task.task_id for task in read_tasks(handle)] == ["mixed"]
    assert handle.meta["filter"]["counts"] == {
        "input": 6,
        "kept": 1,
        "dropped_min_episodes": 0,
        "dropped_lo": 1,
        "dropped_hi": 1,
        "dropped_unsampled": 3,
    }
    assert handle.meta["filter"]["paused_rollout"] == {"n_tasks_sampled": 3, "n_tasks": 6}


def test_allow_paused_is_a_runtime_boolean(taskset: TaskSet, rollouts: Path) -> None:
    with pytest.raises(ConfigError, match="allow_paused must be a boolean"):
        FilterConfig(tasks=str(taskset.root), episodes=str(rollouts), allow_paused="yes")
    plain = FilterConfig(tasks=str(taskset.root), episodes=str(rollouts))
    allowed = replace(plain, allow_paused=True)
    from marli.config import config_hash

    assert config_hash(plain) == config_hash(allowed)
