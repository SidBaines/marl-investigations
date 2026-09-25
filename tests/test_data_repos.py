"""Repository grouping preserves problem bytes, provenance and seeded membership."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from _marli_code_fixtures import code_task

from marli.cli.main import main
from marli.config import compose
from marli.data import repos as repos_module
from marli.data.repos import DataReposConfig
from marli.envs.base import Task
from marli.handles import InputRef
from marli.tasks.taskset import TaskSet, read_tasks, write_tasks
from marli.verbs import run_verb


def source_tasks(root: Path, n: int = 11, *, commit_text: bool = False) -> TaskSet:
    path = write_tasks(
        root,
        (
            replace(code_task(), task_id=f"synthetic/{i}", prompt=f"Synthetic problem {i}.")
            for i in range(n)
        ),
    )
    handle = TaskSet(
        root=root,
        tasks=path.name,
        source="synthetic",
        split="train",
        kind="code",
        n=n,
        answer_format="tests",
        commit_text=commit_text,
    )
    handle.save()
    return handle


async def test_grouping_remainder_manifest_and_streamed_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_tasks(tmp_path / "source")
    original_read, original_write = repos_module.read_tasks, repos_module.write_tasks

    def streamed_read(taskset: TaskSet, *, stream: bool = False) -> Iterator[Task]:
        assert stream
        return original_read(taskset, stream=True)

    def streamed_write(directory: Path, tasks: Iterable[Task]) -> Path:
        assert iter(tasks) is tasks
        return original_write(directory, tasks)

    monkeypatch.setattr(repos_module, "read_tasks", streamed_read)
    monkeypatch.setattr(repos_module, "write_tasks", streamed_write)
    result = await run_verb(
        "data repos", DataReposConfig(tasks=str(source.root), shuffle=False), out=tmp_path / "repos"
    )
    handle = TaskSet.load(result.manifest)
    assert handle == result.handle
    assert handle.kind == "code" and handle.answer_format == "code_repo"
    assert handle.source == source.source and handle.split == source.split
    assert not handle.commit_text
    assert handle.n == handle.meta["n_repos"] == 2
    assert handle.meta["n_per_repo"] == 4
    assert handle.meta["rule_prob"] == 1
    assert handle.meta["families"] == ["header", "constant", "docstring"]
    assert handle.meta["counts"] == {
        "input": 11,
        "kept": 8,
        "dropped_remainder": 3,
        "dropped_max_repos": 0,
        "with_rule": 2,
        "without_rule": 0,
    }
    assert handle.inputs == (InputRef.of(source),)
    assert handle.meta["source"] == asdict(InputRef.of(source))
    manifest = json.loads(result.manifest.read_text())
    assert manifest["kind"] == "taskset" and manifest["tasks"] == "tasks.jsonl"
    assert manifest["inputs"][0]["path"] == str(source.manifest_path)
    repos = list(read_tasks(handle, stream=True))
    originals = read_tasks(source)
    for i, repo in enumerate(repos):
        assert repo.task_id == f"repo-{i:05d}"
        assert repo.prompt and all(task.prompt not in repo.prompt for task in originals)
        assert repo.answer == {
            "kind": "code_repo",
            "problems": [asdict(task) for task in originals[i * 4 : (i + 1) * 4]],
            "has_rule": True,
            "rule_families": ["header", "constant", "docstring"],
        }
    copy = write_tasks(tmp_path / "copy", iter(repos))
    assert copy.read_bytes() == handle.file("tasks").read_bytes()


async def test_shuffle_is_seeded_without_replacement(tmp_path: Path) -> None:
    source = source_tasks(tmp_path / "source", 19)
    cfg = DataReposConfig(tasks=str(source.root), seed=23)
    state = random.getstate()
    outputs = []
    for label, config in (("first", cfg), ("repeat", cfg), ("other", replace(cfg, seed=24))):
        outputs.append((await run_verb("data repos", config, out=tmp_path / label)).handle)
    assert random.getstate() == state
    assert outputs[0].file("tasks").read_bytes() == outputs[1].file("tasks").read_bytes()
    assert outputs[0].file("tasks").read_bytes() != outputs[2].file("tasks").read_bytes()
    expected = read_tasks(source)
    random.Random(23).shuffle(expected)
    actual = [
        problem["task_id"] for repo in read_tasks(outputs[0]) for problem in repo.answer["problems"]
    ]
    assert actual == [task.task_id for task in expected[:16]]
    assert len(set(actual)) == 16


@pytest.mark.parametrize("probability", [0.0, 0.35, 1.0])
async def test_rule_probability_and_families(tmp_path: Path, probability: float) -> None:
    source = source_tasks(tmp_path / "source", 400)
    cfg = DataReposConfig(
        tasks=str(source.root),
        seed=9,
        shuffle=False,
        rule_prob=probability,
        rule_families=("constant",),
    )
    result = await run_verb("data repos", cfg, out=tmp_path / "repos")
    repos = read_tasks(result.handle)
    rng = random.Random(9)
    assert [repo.answer["has_rule"] for repo in repos] == [
        rng.random() < probability for _ in repos
    ]
    assert all(repo.answer["rule_families"] == ["constant"] for repo in repos)
    rate = sum(repo.answer["has_rule"] for repo in repos) / len(repos)
    assert rate == pytest.approx(probability, abs=0.1)
    assert all(
        set(repo.answer) == {"kind", "problems", "has_rule", "rule_families"} for repo in repos
    )


@pytest.mark.parametrize(
    "n_per_repo,max_repos,n,kept,remainder,capped",
    [
        (1, None, 11, 11, 0, 0),
        (4, 1, 1, 4, 3, 4),
        (4, 0, 0, 0, 3, 8),
        (12, None, 0, 0, 11, 0),
    ],
)
async def test_solo_control_caps_and_small_pools(
    tmp_path: Path,
    n_per_repo: int,
    max_repos: int | None,
    n: int,
    kept: int,
    remainder: int,
    capped: int,
) -> None:
    source = source_tasks(tmp_path / "source", commit_text=True)
    result = await run_verb(
        "data repos",
        DataReposConfig(tasks=str(source.root), n_per_repo=n_per_repo, max_repos=max_repos),
        out=tmp_path / "repos",
    )
    assert result.handle.n == n
    assert result.handle.commit_text
    counts = result.handle.meta["counts"]
    assert (counts["kept"], counts["dropped_remainder"], counts["dropped_max_repos"]) == (
        kept,
        remainder,
        capped,
    )
    assert len(list(read_tasks(result.handle, stream=True))) == n


def test_cli_round_trip_idempotency_and_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = source_tasks(tmp_path / "source")
    args = ["data", "repos", f"tasks={source.root}", "--out", str(tmp_path / "repos")]
    for status in ("fresh", "complete"):
        assert main(args) == 0
        output = capsys.readouterr().out
        assert len(output.splitlines()) == 1
        payload = json.loads(output)
        assert payload["ok"] and payload["status"] == status
        assert payload["kind"] == "taskset" and payload["n"] == 2
        assert TaskSet.load(payload["manifest"]).answer_format == "code_repo"
    config = compose(DataReposConfig, overrides=[f"tasks={source.root}", "rule_families=[header]"])
    assert list(config.rule_families) == ["header"]
    with pytest.raises(ValueError, match="typo"):
        compose(DataReposConfig, overrides=[f"tasks={source.root}", "typo=1"])


@pytest.mark.parametrize(
    "config",
    [
        {"tasks": None},
        {"n_per_repo": 0},
        {"n_per_repo": True},
        {"n_per_repo": 1.5},
        {"max_repos": -1},
        {"max_repos": True},
        {"shuffle": 1},
        {"seed": True},
        {"rule_prob": -0.1},
        {"rule_prob": 1.1},
        {"rule_prob": float("nan")},
        {"rule_prob": True},
        {"rule_families": ()},
        {"rule_families": ("typo",)},
    ],
)
def test_invalid_config(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DataReposConfig(**{"tasks": "unused", **config})


async def test_reject_non_code_source(tmp_path: Path) -> None:
    source = source_tasks(tmp_path / "source")
    replace(source, kind="math", answer_format="integer").save()
    with pytest.raises(ValueError, match="code TaskSet"):
        await run_verb(
            "data repos", DataReposConfig(tasks=str(source.root)), out=tmp_path / "repos"
        )
