"""Manifest contracts preserve portable bytes, schema identity, and input provenance."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import FrozenInstanceError, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

import marli.handles as handles_module
from marli import __version__
from marli.errors import ConfigError
from marli.handles import (
    HANDLE_TYPES,
    MANIFEST_VERSION,
    Handle,
    InputRef,
    atomic_write_text,
    inspect_chain,
    load_any,
    register_handle,
    sha256_file,
)


@register_handle
@dataclass(frozen=True, kw_only=True)
class TinyHandle(Handle):
    KIND: ClassVar[str] = "test_task_b_tiny"
    MANIFEST: ClassVar[str] = "tiny.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("data",)
    data: str = "bytes/rows.jsonl"
    n: int = 3


@pytest.mark.parametrize("use_directory", [False, True])
def test_manifest_round_trip(tmp_path: Path, use_directory: bool) -> None:
    source = TinyHandle(root=tmp_path / "source")
    source.save()
    handle = TinyHandle(
        root=tmp_path / "result",
        config_hash="a" * 64,
        inputs=(InputRef.of(source),),
        meta={"label": "α"},
        n=100,
    )
    before = datetime.now(UTC).replace(microsecond=0)
    path = handle.save()
    after = datetime.now(UTC).replace(microsecond=0)
    assert path == handle.root / TinyHandle.MANIFEST == handle.manifest_path
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {
        "kind",
        "manifest_version",
        "marli_version",
        "created_at",
        "config_hash",
        "inputs",
        "meta",
        "data",
        "n",
    }
    assert data["data"] == "bytes/rows.jsonl"
    assert data["n"] == 100
    assert data["kind"] == TinyHandle.KIND
    assert data["manifest_version"] == MANIFEST_VERSION
    assert data["marli_version"] == __version__
    assert data["inputs"] == [
        {"kind": source.KIND, "path": str(source.manifest_path), "sha256": source.sha256()}
    ]
    assert text == json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    created_at = datetime.fromisoformat(data["created_at"])
    assert created_at.tzinfo == UTC
    assert created_at.microsecond == 0
    assert before <= created_at <= after
    loaded = TinyHandle.load(str(handle.root if use_directory else path))
    assert loaded == handle
    assert isinstance(loaded.inputs, tuple)
    assert isinstance(loaded.inputs[0], InputRef)
    assert handle.file("data") == handle.root / "bytes/rows.jsonl"
    assert handle.summary() == {}
    with pytest.raises(FrozenInstanceError):
        loaded.n = 1


def test_absolute_path_field_is_saved_relative_and_survives_move(tmp_path: Path) -> None:
    root = tmp_path / "original"
    data = root / "bytes" / "rows.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"id": 1}\n', encoding="utf-8")
    handle = TinyHandle(root=root, data=str(data))
    handle.save()
    assert json.loads(handle.manifest_path.read_text())["data"] == "bytes/rows.jsonl"
    moved = tmp_path / "moved"
    root.rename(moved)
    loaded = TinyHandle.load(moved)
    assert loaded.root == moved
    assert loaded.file("data").read_text() == '{"id": 1}\n'


@pytest.mark.parametrize("relative", [False, True])
def test_path_field_outside_root_rejected(tmp_path: Path, relative: bool) -> None:
    root = tmp_path / "result"
    data = "../elsewhere.jsonl" if relative else str(tmp_path / "elsewhere.jsonl")
    handle = TinyHandle(root=root, data=data)
    with pytest.raises(ConfigError, match="data.*outside.*root"):
        handle.save()
    assert not handle.manifest_path.exists()


def test_path_field_symlink_outside_root_rejected(tmp_path: Path) -> None:
    root = tmp_path / "result"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ConfigError, match="outside"):
        TinyHandle(root=root, data="escape/rows.jsonl").save()


@pytest.mark.parametrize("name", ["n", "missing", "root"])
def test_file_requires_path_field(tmp_path: Path, name: str) -> None:
    with pytest.raises(ConfigError, match="path field"):
        TinyHandle(root=tmp_path).file(name)


def test_optional_path_field_can_be_absent(tmp_path: Path) -> None:
    @dataclass(frozen=True, kw_only=True)
    class OptionalHandle(Handle):
        KIND: ClassVar[str] = "test_task_b_optional"
        MANIFEST: ClassVar[str] = "optional.json"
        PATH_FIELDS: ClassVar[tuple[str, ...]] = ("state",)
        state: str | None = None

    handle = OptionalHandle(root=tmp_path)
    path = handle.save()
    assert json.loads(path.read_text())["state"] is None
    loaded = OptionalHandle.load(path)
    assert loaded == handle
    with pytest.raises(ConfigError, match="state.*None"):
        loaded.file("state")
    replace(handle, state=str(tmp_path / "state.bin")).save()
    assert OptionalHandle.load(path).file("state") == tmp_path / "state.bin"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"kind": "wrong"}, "expected kind.*wrong"),
        ({"future_field": 3}, "unknown.*future_field"),
        ({"manifest_version": MANIFEST_VERSION + 1}, "manifest_version.*newer"),
    ],
)
def test_load_rejects_schema_drift(tmp_path: Path, updates: dict[str, Any], message: str) -> None:
    path = TinyHandle(root=tmp_path).save()
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, **updates}))
    with pytest.raises(ConfigError, match=message):
        TinyHandle.load(path)


@pytest.mark.parametrize("version", [None, "1", 1.0, True, False, [], {}])
def test_load_rejects_noninteger_manifest_version(tmp_path: Path, version: Any) -> None:
    path = TinyHandle(root=tmp_path).save()
    data = json.loads(path.read_text())
    data["manifest_version"] = version
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError, match="manifest_version") as caught:
        TinyHandle.load(path)
    assert str(path) in str(caught.value)
    assert caught.value.exit_code == 2


def test_load_rejects_missing_manifest_version(tmp_path: Path) -> None:
    path = TinyHandle(root=tmp_path).save()
    data = json.loads(path.read_text())
    del data["manifest_version"]
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError, match="manifest_version") as caught:
        TinyHandle.load(path)
    assert str(path) in str(caught.value)
    assert caught.value.exit_code == 2


@pytest.mark.parametrize("operation", ["load", "save"])
@pytest.mark.parametrize("content", [b"{", b"not JSON", b"\xff", b"[]", b"null", b"1", b'"text"'])
def test_invalid_manifest_is_config_error(tmp_path: Path, operation: str, content: bytes) -> None:
    handle = TinyHandle(root=tmp_path)
    path = handle.manifest_path
    path.write_bytes(content)
    with pytest.raises(ConfigError) as caught:
        if operation == "load":
            TinyHandle.load(path)
        else:
            handle.save()
    assert str(path) in str(caught.value)
    assert caught.value.exit_code == 2
    assert path.read_bytes() == content


def test_load_rejects_missing_required_field(tmp_path: Path) -> None:
    @dataclass(frozen=True, kw_only=True)
    class RequiredHandle(TinyHandle):
        required: str

    path = RequiredHandle(root=tmp_path, required="present").save()
    data = json.loads(path.read_text())
    del data["required"]
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError, match="required") as caught:
        RequiredHandle.load(path)
    assert str(path) in str(caught.value)
    assert caught.value.exit_code == 2


@pytest.mark.parametrize("change", ["extra", "kind", "path", "sha256"])
def test_load_rejects_input_ref_schema_drift(tmp_path: Path, change: str) -> None:
    source = TinyHandle(root=tmp_path / "source")
    source.save()
    path = TinyHandle(root=tmp_path / "result", inputs=(InputRef.of(source),)).save()
    data = json.loads(path.read_text())
    if change == "extra":
        data["inputs"][0]["extra"] = "unexpected"
    else:
        del data["inputs"][0][change]
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError, match=change) as caught:
        TinyHandle.load(path)
    assert str(path) in str(caught.value)
    assert caught.value.exit_code == 2


def test_save_rejects_missing_created_at(tmp_path: Path) -> None:
    handle = TinyHandle(root=tmp_path)
    path = handle.save()
    data = json.loads(path.read_text())
    del data["created_at"]
    content = json.dumps(data)
    path.write_text(content)
    with pytest.raises(ConfigError, match="created_at") as caught:
        handle.save()
    assert str(path) in str(caught.value)
    assert caught.value.exit_code == 2
    assert path.read_text() == content


@pytest.mark.parametrize("use_directory", [False, True])
def test_missing_manifest_suggests_at(tmp_path: Path, use_directory: bool) -> None:
    path = tmp_path if use_directory else tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match=r"TinyHandle\.at\(path\)"):
        TinyHandle.load(path)


@pytest.mark.parametrize("use_directory", [False, True])
def test_ad_hoc_handle(tmp_path: Path, use_directory: bool) -> None:
    data = tmp_path / "rows.jsonl"
    data.write_text("{}\n")
    handle = TinyHandle.at(
        str(tmp_path if use_directory else data), data=data.name, n=1, meta={"source": "manual"}
    )
    assert handle.root == tmp_path
    assert handle.meta == {"source": "manual", "adhoc": True}
    assert handle.n == 1
    assert handle.file("data") == data
    assert not handle.manifest_path.exists()
    assert TinyHandle.at(tmp_path).meta == {"adhoc": True}


def test_ad_hoc_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ad-hoc data not found"):
        TinyHandle.at(tmp_path / "absent")


def test_created_at_survives_load_and_resave(tmp_path: Path) -> None:
    path = TinyHandle(root=tmp_path).save()
    data = json.loads(path.read_text())
    data["created_at"] = "2001-02-03T04:05:06+00:00"
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    loaded = TinyHandle.load(path)
    digest = loaded.sha256()
    assert "created_at" not in loaded.meta
    loaded.save()
    assert loaded.sha256() == digest
    replace(loaded, n=10).save()
    assert json.loads(path.read_text())["created_at"] == data["created_at"]


def test_register_and_load_any(tmp_path: Path) -> None:
    handle = TinyHandle(root=tmp_path)
    handle.save()
    assert HANDLE_TYPES[handle.KIND] is TinyHandle
    assert load_any(str(handle.manifest_path)) == handle
    with pytest.raises(ValueError, match="already registered"):
        register_handle(TinyHandle)
    assert HANDLE_TYPES[handle.KIND] is TinyHandle


def test_load_any_unknown_kind(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text('{"kind": "unregistered_task_b"}')
    with pytest.raises(ConfigError, match="unknown handle kind.*unregistered_task_b"):
        load_any(path)


def test_input_refs_hash_saved_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARLI_DATA", raising=False)
    handle = TinyHandle(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        InputRef.of(handle)
    with pytest.raises(FileNotFoundError):
        handle.sha256()
    path = handle.save()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert handle.sha256() == sha256_file(str(path)) == digest
    ref = InputRef.of(replace(handle, n=1000))
    assert ref == InputRef.from_manifest(str(path))
    assert ref == InputRef(handle.KIND, str(path), digest)
    assert ref.resolve() == path
    with pytest.raises(FrozenInstanceError):
        ref.path = "changed"


def test_portable_input_refs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("MARLI_DATA", str(data_root))
    handle = TinyHandle(root=data_root / "nested" / "result")
    handle.save()
    ref = InputRef.of(handle)
    assert ref == InputRef.from_manifest(handle.manifest_path)
    assert ref.path == "$MARLI_DATA/nested/result/tiny.json"
    assert ref.resolve() == handle.manifest_path
    outside = TinyHandle(root=tmp_path / "data-sibling")
    outside.save()
    outside_ref = InputRef.of(outside)
    assert outside_ref.path == str(outside.manifest_path)
    monkeypatch.setenv("MARLI_DATA", str(tmp_path / "moved"))
    assert ref.resolve() == tmp_path / "moved/nested/result/tiny.json"
    monkeypatch.delenv("MARLI_DATA")
    with pytest.raises(ConfigError, match="MARLI_DATA.*unset"):
        ref.resolve()
    assert outside_ref.resolve() == outside.manifest_path


def make_chain(tmp_path: Path) -> tuple[TinyHandle, TinyHandle, TinyHandle]:
    first = TinyHandle(root=tmp_path / "first", config_hash="first", meta={"n": 1})
    first.save()
    second = TinyHandle(root=tmp_path / "second", inputs=(InputRef.of(first),))
    second.save()
    third = TinyHandle(root=tmp_path / "third", inputs=(InputRef.of(second),))
    third.save()
    return first, second, third


def test_inspect_chain_tampered_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARLI_DATA", str(tmp_path))
    first, second, third = make_chain(tmp_path)
    monkeypatch.delitem(HANDLE_TYPES, TinyHandle.KIND)
    tree = inspect_chain(str(third.manifest_path))
    assert set(tree) == {"kind", "path", "sha256", "config_hash", "meta", "inputs"}
    assert tree["sha256"] == third.sha256()
    middle = tree["inputs"][0]
    leaf = middle["inputs"][0]
    assert middle["path"] == str(second.manifest_path)
    assert middle["sha256_ok"] is True
    assert middle["recorded_sha256"] == second.sha256()
    assert leaf == {
        "kind": first.KIND,
        "path": str(first.manifest_path),
        "sha256": first.sha256(),
        "config_hash": "first",
        "meta": {"n": 1},
        "inputs": [],
        "recorded_sha256": first.sha256(),
        "sha256_ok": True,
    }
    original_sha = first.sha256()
    replace(first, n=99).save()
    leaf = inspect_chain(third.manifest_path)["inputs"][0]["inputs"][0]
    assert leaf["sha256_ok"] is False
    assert leaf["recorded_sha256"] == original_sha
    assert leaf["sha256"] == first.sha256()
    first.manifest_path.unlink()
    leaf = inspect_chain(third.manifest_path)["inputs"][0]["inputs"][0]
    assert leaf == {
        "kind": first.KIND,
        "path": str(first.manifest_path),
        "recorded_sha256": original_sha,
        "missing": True,
    }


def test_inspect_chain_cycle(tmp_path: Path) -> None:
    first, _, third = make_chain(tmp_path)
    replace(first, inputs=(InputRef.of(third),)).save()
    tree = inspect_chain(third.manifest_path)
    assert tree["inputs"][0]["inputs"][0]["inputs"] == [
        {"path": str(third.manifest_path), "cycle": True}
    ]


def test_inspect_chain_checks_each_edge_of_diamond(tmp_path: Path) -> None:
    taskset, checkpoint, evaluation = make_chain(tmp_path)
    original_sha = taskset.sha256()
    replace(taskset, n=99).save()
    replace(evaluation, inputs=(InputRef.of(taskset), InputRef.of(checkpoint))).save()
    tree = inspect_chain(evaluation.manifest_path)
    direct = tree["inputs"][0]
    indirect = tree["inputs"][1]["inputs"][0]
    assert direct["sha256_ok"] is True
    assert direct["recorded_sha256"] == taskset.sha256()
    assert indirect["sha256_ok"] is False
    assert indirect["recorded_sha256"] == original_sha
    assert direct["sha256"] == indirect["sha256"] == taskset.sha256()
    pending = [tree]
    while pending:
        node = pending.pop()
        assert "cycle" not in node
        pending.extend(node["inputs"])


@pytest.mark.parametrize("content", [b"not JSON", b'{"kind":', b"\xff"])
def test_inspect_chain_reports_non_json_input(tmp_path: Path, content: bytes) -> None:
    first, _, third = make_chain(tmp_path)
    original_sha = first.sha256()
    first.manifest_path.write_bytes(content)
    leaf = inspect_chain(third.manifest_path)["inputs"][0]["inputs"][0]
    assert leaf["path"] == str(first.manifest_path)
    assert leaf["invalid"] is True
    assert leaf["sha256_ok"] is False
    assert leaf["recorded_sha256"] == original_sha
    assert leaf["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("existing", [False, True])
def test_atomic_write_respects_umask(tmp_path: Path, existing: bool) -> None:
    path = tmp_path / "file.txt"
    if existing:
        path.write_text("original\n")
        path.chmod(0o644)
    old_umask = os.umask(0o022)
    try:
        atomic_write_text(path, "replacement\n")
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert path.read_text() == "replacement\n"


def test_atomic_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "file.txt"
    atomic_write_text(str(path), "initial\n")
    atomic_write_text(path, "replaced α\n")
    assert path.read_text(encoding="utf-8") == "replaced α\n"
    assert list(tmp_path.rglob("*.tmp*")) == []


def test_atomic_write_failure_preserves_old_file_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file.txt"
    path.write_text("original\n")

    def fail_replace(source: Path, destination: Path) -> None:
        assert source.parent == destination.parent == tmp_path
        assert source.read_text() == "replacement\n"
        assert destination.read_text() == "original\n"
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(handles_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_text(path, "replacement\n")
    assert path.read_text() == "original\n"
    assert list(tmp_path.rglob("*.tmp*")) == []
