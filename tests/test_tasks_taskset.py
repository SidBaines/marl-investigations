"""Task manifests retain domain metadata and relative paths when moved."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest

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
def test_torn_final_line_is_ignored_without_modifying_bytes(tmp_path: Path, tail: bytes) -> None:
    task = Task("synthetic/1", "Synthetic: 1+2?", "3")
    path = write_tasks(tmp_path, [task])
    content = path.read_bytes() + tail
    path.write_bytes(content)
    assert read_tasks(make_taskset(tmp_path)) == [task]
    assert path.read_bytes() == content


def test_corrupt_middle_row_is_loud(tmp_path: Path) -> None:
    path = write_tasks(tmp_path, [Task("synthetic/1", "Synthetic: 1+2?", "3")])
    path.write_bytes(b"broken\n" + path.read_bytes())
    with pytest.raises(json.JSONDecodeError):
        read_tasks(make_taskset(tmp_path))


def test_handle_path_must_stay_relative_to_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside handle root"):
        replace(make_taskset(tmp_path), tasks="../tasks.jsonl").save()
