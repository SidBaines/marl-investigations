"""Python callers share the CLI's input identity and recoverable lifecycle."""

from __future__ import annotations

import dataclasses
import importlib
import logging
import warnings
from pathlib import Path

import _marli_fake_verbs as fake
import pytest

from marli import config, verbs
from marli.errors import ConfigError
from marli.handles import sha256_file
from marli.rundir import RunStatus


@pytest.fixture(autouse=True)
def echo_verb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verbs, "VERBS", {fake.SPEC.name: fake.SPEC})
    monkeypatch.setattr(fake, "invocations", 0)
    monkeypatch.setattr(fake, "crash_after", None)


def test_fake_targets_import_and_resolve() -> None:
    assert importlib.import_module("_marli_fake_verbs") is fake
    assert verbs.resolve(fake.SPEC) == (fake.echo, fake.EchoConfig)
    assert verbs.get_verb("debug echo") is fake.SPEC


async def test_runner_fresh_complete_and_resume(tmp_path: Path, monkeypatch) -> None:
    cfg = fake.EchoConfig()
    first = await verbs.run_verb("debug echo", cfg, out=tmp_path / "fresh")
    assert isinstance(first, verbs.VerbResult)
    assert first.status is RunStatus.FRESH
    assert first.manifest == first.handle.manifest_path == tmp_path / "fresh" / "dummy.json"
    assert first.config_hash == first.handle.config_hash == config.config_hash(cfg)
    assert first.warnings == []
    second = await verbs.run_verb(fake.SPEC, cfg, out=tmp_path / "fresh")
    assert second.status is RunStatus.COMPLETE
    assert second.handle == first.handle
    assert fake.invocations == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.status = RunStatus.COMPLETE
    monkeypatch.setattr(fake, "crash_after", 2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await verbs.run_verb(fake.SPEC, cfg, out=tmp_path / "crashed")
    monkeypatch.setattr(fake, "crash_after", None)
    resumed = await verbs.run_verb(fake.SPEC, cfg, out=tmp_path / "crashed")
    assert resumed.status is RunStatus.RESUME
    assert fake.invocations == 3
    assert len(resumed.handle.file("rows").read_text().splitlines()) == 3


async def test_config_type_is_checked_before_writes(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="requires EchoConfig"):
        await verbs.run_verb(fake.SPEC, {}, out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("field", "target", "message"),
    [
        ("fn", "no_colon", "module:attr"),
        ("fn", ":echo", "module:attr"),
        ("fn", "_marli_fake_verbs:", "module:attr"),
        ("config", "a:b:c", "module:attr"),
        ("fn", "missing_marli_test_module:echo", "cannot resolve"),
        ("fn", "_marli_fake_verbs:absent", "cannot resolve"),
        ("config", "_marli_fake_verbs:absent", "cannot resolve"),
        ("fn", "_marli_fake_verbs:EchoConfig", "coroutine function"),
        ("config", "_marli_fake_verbs:echo", "dataclass type"),
        ("config", "_marli_fake_verbs:config_instance", "dataclass type"),
    ],
)
def test_resolve_bad_targets(field: str, target: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        verbs.resolve(dataclasses.replace(fake.SPEC, **{field: target}))


def test_unknown_verb_lists_known_verbs() -> None:
    with pytest.raises(ConfigError, match="unknown verb.*missing.*known verbs.*debug echo"):
        verbs.get_verb("missing")


@pytest.mark.parametrize("cls", [fake.NestedConfig, fake.OptionalNestedConfig])
async def test_nested_inputs_are_rejected(tmp_path: Path, cls: type) -> None:
    spec = dataclasses.replace(fake.SPEC, config=f"_marli_fake_verbs:{cls.__name__}")
    with pytest.raises(ConfigError, match="nested input field.*child.source.*not supported"):
        await verbs.run_verb(spec, cls(), out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("count", [0, 2])
async def test_input_directory_requires_one_manifest(tmp_path: Path, count: int) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "unrelated.json").write_text('{"kind": "dummy"}')
    (source / "invalid.json").write_text("not json")
    (source / "list.json").write_text("[]")
    for i in range(count):
        (source / f"{i}.json").write_text('{"kind": "dummy", "manifest_version": 1}')
    with pytest.raises(ConfigError, match=f"exactly one handle manifest.*found {count}"):
        await verbs.run_verb(fake.SPEC, fake.EchoConfig(source=str(source)), out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


async def test_missing_input(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="source.*manifest not found"):
        await verbs.run_verb(
            fake.SPEC, fake.EchoConfig(source=str(tmp_path / "missing")), out=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


async def test_input_is_copied_resolved_and_content_hashed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = fake.DummyHandle(root=tmp_path / "source")
    source.save()
    (source.root / "progress.json").write_text('{"done": 3}')
    cfg = fake.EchoConfig(source="source")
    result = await verbs.run_verb(fake.SPEC, cfg, out="result")
    assert cfg.source == "source"
    expected = config.config_hash(
        cfg, input_digests=[f"source={sha256_file(source.manifest_path)}"]
    )
    assert result.config_hash == expected
    direct = await verbs.run_verb(
        fake.SPEC, dataclasses.replace(cfg, source=str(source.manifest_path)), out="result"
    )
    assert direct.status is RunStatus.COMPLETE
    assert direct.config_hash == expected


async def test_warnings_are_captured_and_logged(tmp_path: Path, monkeypatch, caplog) -> None:
    echo = fake.echo

    async def warn(cfg, run):
        warnings.warn("degraded execution", UserWarning, stacklevel=1)
        return await echo(cfg, run)

    monkeypatch.setattr(fake, "echo", warn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with caplog.at_level(logging.WARNING, logger="marli"):
            result = await verbs.run_verb(fake.SPEC, fake.EchoConfig(), out=tmp_path)
    assert result.warnings == ["degraded execution"]
    assert [(r.name, r.levelname, r.message) for r in caplog.records] == [
        ("marli", "WARNING", "degraded execution")
    ]
