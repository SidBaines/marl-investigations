"""Synthetic sources exercise selection and durable CLI output without dataset access."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from marli.cli.main import main
from marli.data import build as build_module
from marli.data.build import BuildConfig
from marli.envs.base import Task
from marli.errors import ConfigError
from marli.handles import InputRef
from marli.registry import Registry
from marli.rundir import RunStatus
from marli.tasks.source import TaskSourceSpec
from marli.tasks.taskset import TaskSet, read_tasks, write_tasks
from marli.verbs import run_verb

type LoaderFixture = tuple[list[dict[str, Any]], list[tuple[str, str | None, str]]]


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LoaderFixture:
    directory = tmp_path / "sources"
    directory.mkdir()
    (directory / "synthetic.yaml").write_text(
        yaml.safe_dump(
            {
                "hf_id": "synthetic/local-only",
                "config": "fixture",
                "split": "train",
                "kind": "math",
                "fields": {"prompt": "question", "answer": "answer"},
                "answer_format": "integer",
                "license": "synthetic",
                "commit_text": False,
            }
        )
    )
    monkeypatch.setattr(build_module, "SOURCES", Registry("tasks", directory, TaskSourceSpec))
    rows = [{"question": f"Synthetic item {i}", "answer": str(i)} for i in range(8)]
    calls: list[tuple[str, str | None, str]] = []

    def loader(hf_id: str, config: str | None, *, split: str) -> list[dict[str, Any]]:
        calls.append((hf_id, config, split))
        return rows

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=loader))
    return rows, calls


def exclusion(tmp_path: Path, prompts: list[str]) -> TaskSet:
    root = tmp_path / "exclude"
    write_tasks(root, [Task(f"excluded/{i}", prompt) for i, prompt in enumerate(prompts)])
    handle = TaskSet(
        root=root,
        tasks="tasks.jsonl",
        source="synthetic-holdout",
        split="test",
        kind="math",
        n=len(prompts),
        answer_format="integer",
        commit_text=False,
    )
    handle.save()
    return handle


async def test_build_manifest_and_rows(tmp_path: Path, source: LoaderFixture) -> None:
    rows, calls = source
    result = await run_verb(
        "data build", BuildConfig(source="synthetic", split="validation"), out=tmp_path / "out"
    )
    handle = TaskSet.load(result.manifest)
    assert handle.summary() == {"n": 8, "source": "synthetic"}
    assert calls == [("synthetic/local-only", "fixture", "validation")]
    assert handle.kind == "math"
    assert handle.split == "validation"
    assert handle.answer_format == "integer"
    assert handle.commit_text is False
    assert handle.inputs == ()
    assert handle.meta == {
        "task_kind": "math",
        "n_raw": 8,
        "n_duplicates": 0,
        "n_conflicting_dropped": 0,
        "counts": {"loaded": 8, "kept": 8, "dropped_exact": 0, "dropped_ngram": 0},
    }
    tasks = read_tasks(handle)
    assert [task.task_id for task in tasks] == [f"synthetic/{i}" for i in range(8)]
    assert [task.prompt for task in tasks] == [row["question"] for row in rows]
    assert [task.answer for task in tasks] == [row["answer"] for row in rows]
    assert all(task.meta["split"] == "validation" for task in tasks)
    manifest = json.loads(result.manifest.read_text())
    assert manifest["kind"] == "taskset"
    assert manifest["tasks"] == "tasks.jsonl"
    assert manifest["config_hash"] == result.config_hash


async def test_source_default_split_and_text_permission(
    tmp_path: Path, source: LoaderFixture
) -> None:
    spec_path = build_module.SOURCES.path("synthetic")
    spec = yaml.safe_load(spec_path.read_text())
    spec["commit_text"] = True
    spec_path.write_text(yaml.safe_dump(spec))
    result = await run_verb("data build", BuildConfig(source="synthetic"), out=tmp_path / "out")
    assert result.handle.commit_text is True
    assert result.handle.split == "train"


async def test_shuffle_and_max_n_are_deterministic(tmp_path: Path, source: LoaderFixture) -> None:
    cfg = BuildConfig(source="synthetic", shuffle=True, seed=19, max_n=4)
    state = random.getstate()
    first = await run_verb("data build", cfg, out=tmp_path / "first")
    second = await run_verb("data build", cfg, out=tmp_path / "second")
    assert random.getstate() == state
    expected = list(range(8))
    random.Random(19).shuffle(expected)
    assert [task.task_id for task in read_tasks(first.handle)] == [
        f"synthetic/{i}" for i in expected[:4]
    ]
    assert first.handle.file("tasks").read_bytes() == second.handle.file("tasks").read_bytes()
    prefix = await run_verb("data build", replace(cfg, shuffle=False), out=tmp_path / "prefix")
    assert [task.task_id for task in read_tasks(prefix.handle)] == [
        f"synthetic/{i}" for i in range(4)
    ]
    empty = await run_verb("data build", replace(cfg, max_n=0), out=tmp_path / "empty")
    assert empty.handle.n == 0
    assert empty.handle.file("tasks").read_bytes() == b""


@pytest.mark.parametrize("ngram_n,kept", [(0, [2, 3, 4, 5]), (2, [3, 4, 5]), (10, [2, 3, 4, 5])])
async def test_decontamination(
    tmp_path: Path, source: LoaderFixture, ngram_n: int, kept: list[int]
) -> None:
    rows, _ = source
    prompts = [
        "red green blue",
        "  RED\n green\tblue  ",
        "prefix GREEN\nblue suffix",
        "blue green",
        "green",
        "blue orange",
    ]
    rows[:] = [{"question": prompt, "answer": "synthetic"} for prompt in prompts]
    excluded = exclusion(tmp_path, ["red green blue", "orange pear"])
    result = await run_verb(
        "data build",
        BuildConfig(source="synthetic", exclude=str(excluded.root), ngram_exclude=ngram_n),
        out=tmp_path / "out",
    )
    handle = result.handle
    assert [task.prompt for task in read_tasks(handle)] == [prompts[i] for i in kept]
    assert handle.meta["decontaminated_against"] == str(excluded.manifest_path)
    assert handle.meta["ngram_exclude"] == ngram_n
    assert handle.meta["counts"] == {
        "loaded": 6,
        "kept": len(kept),
        "dropped_exact": 2,
        "dropped_ngram": 4 - len(kept),
    }
    assert handle.inputs == (InputRef.of(excluded),)
    saved = yaml.safe_load((handle.root / "config.yaml").read_text())
    assert saved["exclude"] == str(excluded.manifest_path)


async def test_max_n_applies_after_exclusion(tmp_path: Path, source: LoaderFixture) -> None:
    excluded = exclusion(tmp_path, ["Synthetic item 0"])
    result = await run_verb(
        "data build",
        BuildConfig(source="synthetic", max_n=2, exclude=str(excluded.manifest_path)),
        out=tmp_path / "out",
    )
    assert [task.task_id for task in read_tasks(result.handle)] == ["synthetic/1", "synthetic/2"]
    assert result.handle.meta["counts"]["loaded"] == 8


def test_build_cli_idempotency(
    tmp_path: Path, source: LoaderFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    _, calls = source
    out = tmp_path / "out"
    args = ["data", "build", "source=synthetic", "max_n=3", "--out", str(out)]
    before: dict[str, bytes] = {}
    for status in ("fresh", "complete"):
        assert main(args) == 0
        captured = capsys.readouterr()
        assert len(captured.out.splitlines()) == 1
        payload = json.loads(captured.out)
        assert payload["ok"] is True
        assert payload["kind"] == "taskset"
        assert payload["n"] == 3
        assert payload["source"] == "synthetic"
        assert payload["status"] == status
        assert payload["manifest"] == str(out / "taskset.json")
        current = {name: (out / name).read_bytes() for name in ("tasks.jsonl", "taskset.json")}
        if before:
            assert current == before
        before = current
    assert len(calls) == 1


async def test_resume_after_task_rows_are_saved(
    tmp_path: Path, source: LoaderFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = BuildConfig(source="synthetic")
    out = tmp_path / "out"
    save = TaskSet.save

    def crash(self: TaskSet) -> Path:
        raise RuntimeError("interrupted before manifest")

    monkeypatch.setattr(TaskSet, "save", crash)
    with pytest.raises(RuntimeError, match="interrupted"):
        await run_verb("data build", cfg, out=out)
    before = (out / "tasks.jsonl").read_bytes()
    assert not (out / "taskset.json").exists()
    monkeypatch.setattr(TaskSet, "save", save)
    resumed = await run_verb("data build", cfg, out=out)
    assert resumed.status is RunStatus.RESUME
    assert resumed.handle.file("tasks").read_bytes() == before
    assert resumed.handle.n == 8


@pytest.mark.parametrize(
    "kwargs", [{"max_n": -1}, {"max_n": True}, {"ngram_exclude": -1}, {"ngram_exclude": 1.5}]
)
def test_invalid_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        BuildConfig(**kwargs)


@pytest.mark.parametrize(
    "prompt", ["Synthetic item 0.", "SYNTHETIC $item 0$!", r"Synthetic \(item 0\)"]
)
async def test_exact_decontamination_ignores_punctuation_and_latex(
    tmp_path: Path,
    source: LoaderFixture,
    prompt: str,
) -> None:
    excluded = exclusion(tmp_path, [prompt])
    result = await run_verb(
        "data build",
        BuildConfig(source="synthetic", max_n=2, exclude=str(excluded.root)),
        out=tmp_path / "out",
    )
    assert [task.task_id for task in read_tasks(result.handle)] == ["synthetic/1", "synthetic/2"]
    assert result.handle.meta["counts"]["dropped_exact"] == 1


async def test_loader_deduplication_metadata_is_retained(
    tmp_path: Path,
    source: LoaderFixture,
) -> None:
    rows, _ = source
    path = build_module.SOURCES.path("synthetic")
    settings = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**settings, "dedupe": True}))
    rows[:] = [
        {"question": "same", "answer": "1"},
        {"question": "same", "answer": "1"},
        {"question": "conflict", "answer": "1"},
        {"question": "conflict", "answer": "2"},
    ]
    result = await run_verb("data build", BuildConfig(source="synthetic"), out=tmp_path / "out")
    assert result.handle.meta["n_raw"] == 4
    assert result.handle.meta["n_duplicates"] == 2
    assert result.handle.meta["n_conflicting_dropped"] == 1
    assert [task.prompt for task in read_tasks(result.handle)] == ["same"]


@pytest.mark.parametrize("field", ["max_tests", "max_test_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_code_build_caps(field: str, value: Any) -> None:
    with pytest.raises(ConfigError, match=field):
        BuildConfig(**{field: value})


async def test_math_rejects_code_build_caps(tmp_path: Path, source: LoaderFixture) -> None:
    with pytest.raises(ConfigError, match="code source"):
        await run_verb(
            "data build", BuildConfig(source="synthetic", max_tests=4), out=tmp_path / "out"
        )


async def test_code_build_records_overrides_counts_and_streams_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from functools import partial

    from _marli_code_fixtures import many_tests_row

    from marli.tasks.loaders import load_tasks

    def rows(*args: Any, **kwargs: Any) -> Any:
        for i in range(4):
            yield {**many_tests_row("deepcoder", public=False), "problem": f"Synthetic {i}"}

    monkeypatch.setattr(build_module, "load_tasks", partial(load_tasks, loader=rows))
    writer = build_module.write_tasks

    def streamed(directory: Path, tasks: Any) -> Path:
        assert iter(tasks) is tasks
        return writer(directory, tasks)

    monkeypatch.setattr(build_module, "write_tasks", streamed)
    built = await run_verb(
        "data build",
        BuildConfig(source="deepcoder", max_n=2, max_tests=2, max_test_bytes=3),
        out=tmp_path / "built",
    )
    handle = TaskSet.load(built.manifest)
    assert handle.n == 2
    assert handle.meta["max_tests"] == 2 and handle.meta["max_test_bytes"] == 3
    # 3 subsets, 4 unique prompts: only first parseable rows contribute test drops.
    assert handle.meta["n_raw"] == 12 and handle.meta["n_duplicates"] == 8
    assert handle.meta["n_tests_dropped_bytes"] == 160
    assert handle.meta["n_tests_dropped_cap"] == 32
    assert handle.meta["counts"]["loaded"] == 4
    assert all(task.meta["n_tests"] == 2 for task in read_tasks(handle))
