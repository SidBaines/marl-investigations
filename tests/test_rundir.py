"""Run ownership and recovery must preserve completed work across interruptions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pytest

import marli.rundir as rundir_module
from marli.config import config_hash
from marli.errors import ConfigError, HashMismatchError, MarliError
from marli.handles import Handle
from marli.rundir import RunDir, RunDirLockedError, RunStatus


@dataclass(frozen=True, kw_only=True)
class RowsHandle(Handle):
    KIND: ClassVar[str] = "test_task_b_rows"
    MANIFEST: ClassVar[str] = "rows.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("rows",)
    rows: str = "rows.jsonl"


@dataclass(frozen=True)
class Settings:
    seed: int = 1


def run_dir(out: Path, *, seed: int = 1, force: bool = False) -> RunDir:
    return RunDir(
        str(out),
        kind=RowsHandle.KIND,
        manifest_name=RowsHandle.MANIFEST,
        config_hash=config_hash(Settings(seed)),
        force=force,
    )


def test_fresh_rows_finalize_and_complete(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with run_dir(out) as run:
        assert run.out == out.resolve()
        assert run.status is RunStatus.FRESH
        assert not run.path(RowsHandle.MANIFEST).exists()
        assert run.read_rows("absent.jsonl") == []
        assert run.done_keys("absent.jsonl", "id") == set()
        rows = [{"id": 1, "answer": "α"}, {"id": 2, "answer": "β"}]
        for row in rows:
            run.append_row("rows.jsonl", MappingProxyType(row))
        assert run.read_rows("rows.jsonl") == rows
        assert run.done_keys("rows.jsonl", "id") == {1, 2}
        assert run.path("rows.jsonl").read_text() == (
            '{"answer": "α", "id": 1}\n{"answer": "β", "id": 2}\n'
        )
        handle = RowsHandle(root=out, config_hash="unsaved hash")
        saved = run.finalize(handle)
        assert saved is not handle
        assert saved.config_hash == run.config_hash
        assert handle.config_hash == "unsaved hash"
        assert RowsHandle.load(out) == saved
        assert run.status is RunStatus.COMPLETE
        manifest_bytes = saved.manifest_path.read_bytes()
        record_bytes = run.path(".marli/run.json").read_bytes()
    with run_dir(out) as reopened:
        assert reopened.status is RunStatus.COMPLETE
        assert reopened.read_rows("rows.jsonl") == rows
        assert reopened.path(RowsHandle.MANIFEST).read_bytes() == manifest_bytes
        assert reopened.path(".marli/run.json").read_bytes() == record_bytes
    assert list(out.rglob("*.tmp*")) == []


def test_fresh_run_record_uses_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provenance = {"git_commit": "abc", "git_dirty": True, "host": "test"}

    def fake_provenance() -> dict[str, Any]:
        return provenance

    monkeypatch.setattr(rundir_module.runlog, "provenance", fake_provenance)
    before = datetime.now(UTC).replace(microsecond=0)
    with run_dir(tmp_path) as run:
        record = json.loads(run.path(".marli/run.json").read_text())
        assert set(record) == {"kind", "config_hash", "manifest", "provenance", "started_at"}
        assert record["kind"] == RowsHandle.KIND
        assert record["config_hash"] == config_hash(Settings())
        assert record["manifest"] == RowsHandle.MANIFEST
        assert record["provenance"] == provenance
    after = datetime.now(UTC).replace(microsecond=0)
    started_at = datetime.fromisoformat(record["started_at"])
    assert started_at.tzinfo == UTC
    assert started_at.microsecond == 0
    assert before <= started_at <= after


@pytest.mark.parametrize("complete", [False, True])
def test_hash_mismatch_releases_lock(tmp_path: Path, complete: bool) -> None:
    with run_dir(tmp_path) as original:
        if complete:
            original.finalize(RowsHandle(root=tmp_path))
    mismatched = run_dir(tmp_path, seed=2)
    with pytest.raises(HashMismatchError) as caught:
        mismatched.open()
    assert caught.value.exit_code == 3
    message = str(caught.value)
    assert original.config_hash[:12] in message
    assert mismatched.config_hash[:12] in message
    assert "--force" in message and "--out" in message
    with run_dir(tmp_path) as reopened:
        assert reopened.status is (RunStatus.COMPLETE if complete else RunStatus.RESUME)


def test_crash_resumes_without_rewriting_record(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="simulated crash"), run_dir(tmp_path) as run:
        run.append_row("nested/rows.jsonl", {"id": "done"})
        record = run.path(".marli/run.json").read_bytes()
        raise RuntimeError("simulated crash")
    assert not run.path(RowsHandle.MANIFEST).exists()
    with run_dir(tmp_path) as resumed:
        assert resumed.status is RunStatus.RESUME
        assert resumed.path(".marli/run.json").read_bytes() == record
        assert resumed.read_rows("nested/rows.jsonl") == [{"id": "done"}]
        assert resumed.done_keys("nested/rows.jsonl", "id") == {"done"}


@pytest.mark.parametrize(
    "tail", [b'{"id":', b'{"id": 2}', b"invalid\n", b'{"answer": "\xce', b"\xff\n"]
)
@pytest.mark.parametrize("with_complete_row", [False, True])
def test_torn_final_line_is_truncated_before_append(
    tmp_path: Path, tail: bytes, with_complete_row: bool
) -> None:
    with run_dir(tmp_path) as run:
        prefix = '{"answer": "α", "id": 1}\n'.encode() if with_complete_row else b""
        path = run.path("rows.jsonl")
        path.write_bytes(prefix + tail)
        rows = [{"answer": "α", "id": 1}] if with_complete_row else []
        assert run.read_rows("rows.jsonl") == rows
        assert path.read_bytes() == prefix
        run.append_row("rows.jsonl", {"id": 3})
        assert run.read_rows("rows.jsonl") == [*rows, {"id": 3}]
        assert path.read_bytes() == prefix + b'{"id": 3}\n'


@pytest.mark.parametrize("tail", [b'{"id":', b'{"id": 2}', b'{"answer": "\xce'])
def test_resume_appends_before_reading_torn_tail(tmp_path: Path, tail: bytes) -> None:
    with run_dir(tmp_path) as run:
        run.append_row("rows.jsonl", {"id": 1})
        path = run.path("rows.jsonl")
        path.write_bytes(path.read_bytes() + tail)
    with run_dir(tmp_path) as resumed:
        resumed.append_row("rows.jsonl", {"id": 3})
        resumed.append_row("rows.jsonl", {"id": 4})
        expected = [{"id": 1}, {"id": 3}, {"id": 4}]
        assert resumed.read_rows("rows.jsonl") == expected
        assert path.read_bytes().endswith(b"\n")
        assert [json.loads(line) for line in path.read_bytes().splitlines()] == expected


def test_append_with_torn_tail_rejects_middle_corruption(tmp_path: Path) -> None:
    with run_dir(tmp_path) as run:
        path = run.path("rows.jsonl")
        content = b'{"id": 1}\ninvalid\n{"id":'
        path.write_bytes(content)
        with pytest.raises(MarliError, match="line 2"):
            run.append_row("rows.jsonl", {"id": 3})
        assert path.read_bytes() == content


@pytest.mark.parametrize("bad_line", [b"invalid\n", b"\xff\n"])
def test_corrupt_middle_line_is_loud_and_untouched(tmp_path: Path, bad_line: bytes) -> None:
    with run_dir(tmp_path) as run:
        path = run.path("rows.jsonl")
        content = b'{"id": 1}\n' + bad_line + b'{"id": 3}\n'
        path.write_bytes(content)
        with pytest.raises(MarliError) as caught:
            run.read_rows("rows.jsonl")
        assert str(path) in str(caught.value)
        assert "line 2" in str(caught.value)
        assert path.read_bytes() == content


@pytest.mark.parametrize("complete", [False, True])
def test_force_wipes_old_work_but_preserves_lock(tmp_path: Path, complete: bool) -> None:
    with run_dir(tmp_path) as run:
        run.append_row("nested/rows.jsonl", {"id": 1})
        run.write_progress({"done": 1})
        run.path(".marli/stale").mkdir()
        run.path(".marli/stale/data").write_text("old")
        if complete:
            run.finalize(RowsHandle(root=tmp_path))
        inode = run.path(".marli/lock").stat().st_ino
    with run_dir(tmp_path, seed=2, force=True) as forced:
        assert forced.status is RunStatus.FRESH
        assert forced.path(".marli/lock").stat().st_ino == inode
        assert {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")} == {
            ".marli",
            ".marli/lock",
            ".marli/run.json",
        }
        with pytest.raises(RunDirLockedError):
            run_dir(tmp_path, force=True).open()
        record = json.loads(forced.path(".marli/run.json").read_text())
        assert record["config_hash"] == config_hash(Settings(2))


def test_interrupted_force_cannot_reuse_old_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_dir(tmp_path) as run:
        run.append_row("rows.jsonl", {"id": 1})
        run.finalize(RowsHandle(root=tmp_path))
    manifest = run.path(RowsHandle.MANIFEST)
    record = run.path(".marli/run.json")
    real_fsync = os.fsync
    out_synced = False
    removed: list[Path] = []

    def track_fsync(fd: int) -> None:
        nonlocal out_synced
        if os.fstat(fd).st_ino == tmp_path.stat().st_ino:
            assert not manifest.exists()
            assert not record.exists()
            out_synced = True
        real_fsync(fd)

    def interrupt_remove(path: Path) -> None:
        assert not manifest.exists()
        assert not record.exists()
        assert out_synced
        removed.append(path)
        raise KeyboardInterrupt

    monkeypatch.setattr(rundir_module.os, "fsync", track_fsync)
    monkeypatch.setattr(RunDir, "_remove", staticmethod(interrupt_remove))
    with pytest.raises(KeyboardInterrupt):
        run_dir(tmp_path, seed=2, force=True).open()
    assert len(removed) == 1
    assert not manifest.exists()
    assert not record.exists()
    assert run.path("rows.jsonl").read_bytes() == b'{"id": 1}\n'
    with pytest.raises(ConfigError, match="not a marli run dir"):
        run_dir(tmp_path).open()


def test_force_unlinks_symlinks_without_deleting_targets(tmp_path: Path) -> None:
    outside = tmp_path / "keep"
    outside.mkdir()
    (outside / "valuable").write_text("keep")
    out = tmp_path / "run"
    out.mkdir()
    (out / "linked").symlink_to(outside, target_is_directory=True)
    with run_dir(out, force=True) as forced:
        assert forced.status is RunStatus.FRESH
        assert not forced.path("linked").exists()
        assert (outside / "valuable").read_text() == "keep"


def test_foreign_dir_refused_and_force_allowed(tmp_path: Path) -> None:
    data = tmp_path / "valuable.txt"
    data.write_text("keep me")
    with pytest.raises(ConfigError, match="not a marli run dir.*--force.*empty dir"):
        run_dir(tmp_path).open()
    assert data.read_text() == "keep me"
    with run_dir(tmp_path, force=True) as forced:
        assert forced.status is RunStatus.FRESH
        assert not data.exists()


def test_exclusive_lock_then_reopen(tmp_path: Path) -> None:
    first = run_dir(tmp_path)
    second = run_dir(tmp_path)
    try:
        assert first.open() is RunStatus.FRESH
        assert first.open() is RunStatus.FRESH
        with pytest.raises(RunDirLockedError) as caught:
            second.open()
        assert caught.value.exit_code == 1
        assert str(tmp_path) in str(caught.value)
        with pytest.raises(RunDirLockedError):
            run_dir(tmp_path).open()
        first.close()
        assert second.open() is RunStatus.RESUME
    finally:
        first.close()
        second.close()


def test_forked_child_close_preserves_parent_lock(tmp_path: Path) -> None:
    with run_dir(tmp_path) as parent:
        pid = os.fork()
        if pid == 0:
            try:
                parent.close()
            except BaseException:
                os._exit(1)
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        with pytest.raises(RunDirLockedError), run_dir(tmp_path):
            pass
    with run_dir(tmp_path) as reopened:
        assert reopened.status is RunStatus.RESUME


@pytest.mark.parametrize("escape", ["absolute", "absolute_inside", "parent", "symlink"])
def test_paths_cannot_escape_run(tmp_path: Path, escape: str) -> None:
    outside = tmp_path / "outside.jsonl"
    content = b'{"id": 1}\n{"id":'
    outside.write_bytes(content)
    with run_dir(tmp_path / "run") as run:
        run.path("linked").symlink_to(tmp_path, target_is_directory=True)
        rel = {
            "absolute": str(outside),
            "absolute_inside": str(run.out / "rows.jsonl"),
            "parent": "../outside.jsonl",
            "symlink": "linked/outside.jsonl",
        }[escape]
        with pytest.raises(ConfigError):
            run.path(rel)
        with pytest.raises(ConfigError):
            run.read_rows(rel)
        with pytest.raises(ConfigError):
            run.append_row(rel, {"id": 2})
        assert outside.read_bytes() == content


def test_resume_kind_mismatch(tmp_path: Path) -> None:
    with run_dir(tmp_path):
        pass
    wrong_kind = RunDir(
        tmp_path,
        kind="wrong",
        manifest_name=RowsHandle.MANIFEST,
        config_hash=config_hash(Settings()),
    )
    with pytest.raises(HashMismatchError, match="kind.*wrong"):
        wrong_kind.open()
    with run_dir(tmp_path) as resumed:
        assert resumed.status is RunStatus.RESUME


def test_finalize_requires_same_root(tmp_path: Path) -> None:
    with run_dir(tmp_path / "run") as run:
        with pytest.raises(ConfigError, match="root.*does not match"):
            run.finalize(RowsHandle(root=tmp_path / "elsewhere"))
        assert run.status is RunStatus.FRESH
        assert not run.path(RowsHandle.MANIFEST).exists()


@pytest.mark.parametrize("after_close", [False, True])
def test_methods_require_open(tmp_path: Path, after_close: bool) -> None:
    run = run_dir(tmp_path)
    if after_close:
        run.open()
        run.close()
    assert run.path("data") == tmp_path / "data"
    calls = [
        (run.append_row, ("rows.jsonl", {"id": 1})),
        (run.read_rows, ("rows.jsonl",)),
        (run.done_keys, ("rows.jsonl", "id")),
        (run.write_progress, ({"done": 1},)),
        (run.finalize, (RowsHandle(root=tmp_path),)),
    ]
    for method, args in calls:
        with pytest.raises(MarliError, match=r"call open\(\) first"):
            method(*args)
    run.close()


def test_write_progress(tmp_path: Path) -> None:
    before = datetime.now(UTC).replace(microsecond=0)
    with run_dir(tmp_path) as run:
        run.write_progress({"done": 1, "updated_at": "old", "label": "α"})
        run.write_progress(MappingProxyType({"done": 2, "updated_at": "old"}))
        progress = json.loads(run.path("progress.json").read_text())
        assert set(progress) == {"done", "updated_at"}
        assert progress["done"] == 2
    after = datetime.now(UTC).replace(microsecond=0)
    timestamp = datetime.fromisoformat(progress["updated_at"])
    assert timestamp.tzinfo == UTC
    assert before <= timestamp <= after
    assert list(tmp_path.rglob("*.tmp*")) == []


@pytest.mark.parametrize("reopen", [False, True])
def test_completed_run_rejects_rows_and_progress(tmp_path: Path, reopen: bool) -> None:
    with run_dir(tmp_path) as run:
        run.append_row("rows.jsonl", {"id": 1})
        run.write_progress({"done": 1})
        run.finalize(RowsHandle(root=tmp_path))
        paths = [run.path(name) for name in ("rows.jsonl", "progress.json", "rows.json")]
        original = [path.read_bytes() for path in paths]
        if reopen:
            run.close()
            run.open()
        with pytest.raises(MarliError, match="complete"):
            run.append_row("rows.jsonl", {"id": 2})
        with pytest.raises(MarliError, match="complete"):
            run.append_row("new/rows.jsonl", {"id": 2})
        with pytest.raises(MarliError, match="complete"):
            run.write_progress({"done": 2})
        assert [path.read_bytes() for path in paths] == original
        assert not run.path("new").exists()


def test_failed_open_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_write = rundir_module.atomic_write_text

    def fail_write(path: str | Path, text: str) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(rundir_module, "atomic_write_text", fail_write)
    with pytest.raises(OSError, match="simulated disk failure"):
        run_dir(tmp_path).open()
    monkeypatch.setattr(rundir_module, "atomic_write_text", real_write)
    with run_dir(tmp_path) as recovered:
        assert recovered.status is RunStatus.FRESH
