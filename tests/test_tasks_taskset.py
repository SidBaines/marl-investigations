"""Task manifests retain domain metadata and relative paths when moved."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Any

import pytest
from _marli_code_fixtures import code_task

from marli.envs.base import Task
from marli.handles import HANDLE_TYPES, InputRef, load_any
from marli.tasks.taskset import TaskSet, read_tasks, write_tasks


def make_taskset(root: Path, n: int = 2) -> TaskSet:
    return TaskSet(
        root=root,
        tasks="tasks.jsonl",
        source="synthetic",
        split="train",
        kind="math",
        n=n,
        answer_format="latex",
        commit_text=False,
        meta={"fixture": True},
    )


def test_jsonl_and_manifest_roundtrip_after_move(tmp_path: Path) -> None:
    tasks = [
        Task("synthetic/1", "Synthetic: compute 2 + 3.\nExplain α.", "5", {"row_index": 0}),
        Task("synthetic/2", "Synthetic: divide 3 by 4.", r"\frac{3}{4}", {"difficulty": 0.875}),
    ]
    root = tmp_path / "original"
    path = write_tasks(root, iter(tasks))
    assert path == root / "tasks.jsonl"
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        asdict(t) for t in tasks
    ]
    handle = make_taskset(root)
    manifest = handle.save()
    raw = json.loads(manifest.read_text())
    assert raw["kind"] == "taskset"
    assert raw["meta"] == {"fixture": True, "task_kind": "math"}
    assert raw["tasks"] == "tasks.jsonl"
    assert raw["commit_text"] is False
    assert HANDLE_TYPES["taskset"] is TaskSet
    assert load_any(manifest) == TaskSet.load(root) == handle
    assert read_tasks(handle) == tasks
    assert handle.summary() == {"n": 2, "source": "synthetic"}
    ref = InputRef.of(handle)
    assert ref.kind == "taskset"
    moved = tmp_path / "moved"
    root.rename(moved)
    loaded = TaskSet.load(moved)
    assert loaded.kind == "math"
    assert read_tasks(loaded) == tasks
    assert loaded.sha256() == ref.sha256
    with pytest.raises(FrozenInstanceError):
        loaded.n = 99  # type: ignore[misc]


def test_empty_taskset_and_replacement(tmp_path: Path) -> None:
    write_tasks(tmp_path, [Task("synthetic/1", "Synthetic: 1+2?", "3")])
    write_tasks(str(tmp_path), [])
    handle = make_taskset(tmp_path, n=0)
    handle.save()
    assert read_tasks(TaskSet.load(handle.manifest_path)) == []
    assert handle.file("tasks").read_bytes() == b""


@pytest.mark.parametrize("tail", [b'{"task_id":', b'{"prompt":"\xce', b"broken\n", b"{}"])
def test_corrupt_final_line_is_loud_without_modifying_bytes(tmp_path: Path, tail: bytes) -> None:
    task = Task("synthetic/1", "Synthetic: 1+2?", "3")
    path = write_tasks(tmp_path, [task])
    content = path.read_bytes() + tail
    path.write_bytes(content)
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError, TypeError)):
        read_tasks(make_taskset(tmp_path))
    assert path.read_bytes() == content


def test_valid_final_line_without_newline_is_read(tmp_path: Path) -> None:
    tasks = [Task("synthetic/1", "Synthetic: 1+2?", "3"), Task("synthetic/2", "Synthetic α", "4")]
    path = write_tasks(tmp_path, tasks)
    content = path.read_bytes().removesuffix(b"\n")
    path.write_bytes(content)
    assert read_tasks(make_taskset(tmp_path)) == tasks
    assert path.read_bytes() == content


@pytest.mark.parametrize("n", [0, 2])
def test_manifest_count_mismatch_is_loud(tmp_path: Path, n: int) -> None:
    path = write_tasks(tmp_path, [Task("synthetic/1", "Synthetic", "3")])
    content = path.read_bytes()
    with pytest.raises(ValueError, match=f"row count mismatch: expected {n}, read 1"):
        read_tasks(make_taskset(tmp_path, n=n))
    assert path.read_bytes() == content


@pytest.mark.parametrize(
    "changes",
    [{"n": -1}, {"n": 1.5}, {"n": True}, {"answer_format": "float"}, {"commit_text": "false"}],
)
def test_invalid_taskset_fields_rejected_on_construction_and_load(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    handle = make_taskset(tmp_path)
    with pytest.raises(ValueError):
        replace(handle, **changes)
    manifest = handle.save()
    raw = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**raw, **changes}))
    with pytest.raises(ValueError):
        TaskSet.load(manifest)


def test_corrupt_middle_row_is_loud(tmp_path: Path) -> None:
    path = write_tasks(tmp_path, [Task("synthetic/1", "Synthetic: 1+2?", "3")])
    path.write_bytes(b"broken\n" + path.read_bytes())
    with pytest.raises(json.JSONDecodeError):
        read_tasks(make_taskset(tmp_path))


def test_handle_path_must_stay_relative_to_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside handle root"):
        replace(make_taskset(tmp_path), tasks="../tasks.jsonl").save()


@pytest.mark.parametrize("functional", [False, True])
def test_code_taskset_roundtrips_large_bundles(tmp_path: Path, functional: bool) -> None:
    task = code_task(functional=functional)
    task.answer["tests"].append({"input": "1\n" * 100_000, "output": "2" * 100_000})
    root = tmp_path / "original"
    write_tasks(root, [task])
    handle = replace(make_taskset(root, n=1), kind="code", answer_format="tests")
    handle.save()
    raw = json.loads(handle.manifest_path.read_text())
    assert raw["kind"] == "taskset" and raw["meta"]["task_kind"] == "code"
    assert "answer" not in raw and handle.manifest_path.stat().st_size < 2048
    moved = tmp_path / "moved"
    root.rename(moved)
    loaded = TaskSet.load(moved)
    assert loaded.kind == "code" and loaded.answer_format == "tests"
    assert read_tasks(loaded) == [task]


@pytest.mark.parametrize(
    "kind,answer_format",
    [("math", "tests"), ("code", "integer"), ("code", "latex"), ("bad", "tests")],
)
def test_taskset_domain_and_answer_format_agree(
    tmp_path: Path, kind: str, answer_format: str
) -> None:
    with pytest.raises(ValueError):
        replace(make_taskset(tmp_path), kind=kind, answer_format=answer_format)
