"""Python callers share the CLI's input identity and recoverable lifecycle."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace

import _marli_fake_verbs as fake
import pytest

from marli import config, verbs
from marli.errors import ConfigError, MarliError
from marli.handles import Handle, sha256_file
from marli.rundir import RunDir, RunStatus


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
async def test_nested_inputs_are_resolved(
    tmp_path: Path, cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = fake.DummyHandle(root=tmp_path / "source")
    source.save()
    spec = dataclasses.replace(fake.SPEC, config=f"_marli_fake_verbs:{cls.__name__}")

    async def nested(cfg: fake.NestedConfig | fake.OptionalNestedConfig, run: RunDir) -> Handle:
        assert cfg.child is not None
        assert cfg.child.source == str(source.manifest_path)
        return fake.DummyHandle(root=run.out)

    monkeypatch.setattr(fake, "echo", nested)
    cfg = cls(child=fake.EchoConfig(source=str(source.root)))
    result = await verbs.run_verb(spec, cfg, out=tmp_path / "out")
    assert result.config_hash == config.config_hash(
        cfg, input_digests=[f"child.source={source.sha256()}"]
    )


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
    # Pytest replaces showwarning for each test; restore the process-wide hook.
    monkeypatch.setattr(warnings, "showwarning", verbs._showwarning)
    echo = fake.echo

    async def warn(cfg, run):
        warnings.warn("degraded execution", UserWarning, stacklevel=1)
        return await echo(cfg, run)

    monkeypatch.setattr(fake, "echo", warn)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING, logger="marli"):
            result = await verbs.run_verb(fake.SPEC, fake.EchoConfig(), out=tmp_path)
    assert result.warnings == ["degraded execution"]
    assert [(r.name, r.levelname, r.message) for r in caplog.records] == [
        ("marli", "WARNING", "degraded execution")
    ]


@pytest.mark.parametrize("returned", ["none", "lookalike", "base_handle", "wrong_manifest"])
async def test_wrong_handles_are_rejected_before_finalizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returned: str
) -> None:
    spec = fake.SPEC
    if returned == "wrong_manifest":
        spec = dataclasses.replace(spec, manifest="expected.json")
        value = fake.DummyHandle(root=tmp_path)
    elif returned == "lookalike":
        value = SimpleNamespace(MANIFEST=spec.manifest)
    elif returned == "base_handle":
        value = Handle(root=tmp_path)
    else:
        value = None

    async def wrong(cfg: fake.EchoConfig, run: RunDir) -> object:
        return value

    monkeypatch.setattr(fake, "echo", wrong)
    with pytest.raises(MarliError) as caught:
        await verbs.run_verb(spec, fake.EchoConfig(), out=tmp_path)
    message = str(caught.value)
    assert spec.name in message
    assert spec.manifest in message
    assert type(value).__name__ in message
    assert repr(getattr(value, "MANIFEST", None)) in message
    assert not list(tmp_path.glob("*.json"))


async def test_concurrent_warning_collectors_preserve_filters_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(warnings, "showwarning", verbs._showwarning)
    echo = fake.echo
    first_ready, second_ready = asyncio.Event(), asyncio.Event()

    async def warn(cfg: fake.EchoConfig, run: RunDir) -> Handle:
        if cfg.message == "second":
            await first_ready.wait()
        warnings.warn(f"{cfg.message} one", stacklevel=1)
        if cfg.message == "first":
            first_ready.set()
            await second_ready.wait()
        else:
            second_ready.set()
            await asyncio.sleep(0)
        warnings.warn(f"{cfg.message} two", stacklevel=1)
        warnings.warn(f"{cfg.message} one", stacklevel=1)
        return await echo(cfg, run)

    monkeypatch.setattr(fake, "echo", warn)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        filters = warnings.filters
        before = filters[:]
        first, second = await asyncio.gather(
            verbs.run_verb(fake.SPEC, fake.EchoConfig(message="first"), out=tmp_path / "first"),
            verbs.run_verb(fake.SPEC, fake.EchoConfig(message="second"), out=tmp_path / "second"),
        )
        assert warnings.filters is filters
        assert warnings.filters == before
    assert first.warnings == ["first one", "first two"]
    assert second.warnings == ["second one", "second two"]


@pytest.mark.parametrize("fail", [False, True])
async def test_warning_hook_delegates_after_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    monkeypatch.setattr(warnings, "showwarning", verbs._showwarning)
    delegated: list[str] = []

    def original(message: Warning | str, *args: object) -> None:
        delegated.append(str(message))

    echo = fake.echo

    async def warn(cfg: fake.EchoConfig, run: RunDir) -> Handle:
        warnings.warn("inside", stacklevel=1)
        if fail:
            raise RuntimeError("failure")
        return await echo(cfg, run)

    monkeypatch.setattr(verbs, "_original_showwarning", original)
    monkeypatch.setattr(fake, "echo", warn)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("before", stacklevel=1)
        if fail:
            with pytest.raises(RuntimeError, match="failure"):
                await verbs.run_verb(fake.SPEC, fake.EchoConfig(), out=tmp_path)
        else:
            result = await verbs.run_verb(fake.SPEC, fake.EchoConfig(), out=tmp_path)
            assert result.warnings == ["inside"]
        warnings.warn("after", stacklevel=1)
    assert delegated == ["before", "after"]


async def test_warning_capture_respects_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(warnings, "showwarning", verbs._showwarning)
    echo = fake.echo

    async def warn(cfg: fake.EchoConfig, run: RunDir) -> Handle:
        warnings.warn("ignored", stacklevel=1)
        return await echo(cfg, run)

    monkeypatch.setattr(fake, "echo", warn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = warnings.filters[:]
        result = await verbs.run_verb(fake.SPEC, fake.EchoConfig(), out=tmp_path)
        assert warnings.filters == before
    assert result.warnings == []


async def test_path_input_preserves_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = fake.DummyHandle(root=tmp_path / "source")
    source.save()
    monkeypatch.chdir(tmp_path)
    cfg = fake.PathConfig(source=Path("source"))
    spec = dataclasses.replace(fake.SPEC, config="_marli_fake_verbs:PathConfig")

    async def check_path(cfg: fake.PathConfig, run: RunDir) -> Handle:
        assert isinstance(cfg.source, Path)
        assert cfg.source == source.manifest_path
        return fake.DummyHandle(root=run.out)

    monkeypatch.setattr(fake, "echo", check_path)
    await verbs.run_verb(spec, cfg, out=tmp_path / "out")
    assert cfg.source == Path("source")


@pytest.mark.parametrize("operation", ["stat", "read_text"])
def test_manifest_search_skips_large_and_unreadable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    source = fake.DummyHandle(root=tmp_path)
    source.save()
    large = tmp_path / "large.json"
    large.write_text('{"kind":"dummy","manifest_version":1,"data":"' + "x" * 1_000_000 + '"}')
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text('{"kind":"dummy","manifest_version":1}')
    original = getattr(Path, operation)

    def fail(path: Path, *args: object, **kwargs: object) -> object:
        if path == unreadable:
            raise OSError("unreadable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, fail)
    assert verbs.input_manifest(tmp_path, "source") == source.manifest_path


@dataclasses.dataclass
class CollectionInputs:
    children: list[fake.EchoConfig] = dataclasses.field(default_factory=list)
    mapping: dict[str, fake.EchoConfig] = dataclasses.field(default_factory=dict)


async def test_nested_input_digests_cover_lists_dicts_and_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marli.errors import HashMismatchError

    source = fake.DummyHandle(root=tmp_path / "source")
    source.save()
    cfg = CollectionInputs(
        [fake.EchoConfig(source=str(source.root))],
        {"a": fake.EchoConfig(source=str(source.manifest_path))},
    )

    async def echo(cfg: CollectionInputs, run: RunDir) -> Handle:
        assert cfg.children[0].source == cfg.mapping["a"].source == str(source.manifest_path)
        return fake.DummyHandle(root=run.out)

    monkeypatch.setattr(verbs, "resolve", lambda spec: (echo, CollectionInputs))
    result = await verbs.run_verb(fake.SPEC, cfg, out=tmp_path / "out")
    digest = sha256_file(source.manifest_path)
    assert result.config_hash == config.config_hash(
        cfg,
        input_digests=[
            f"children[0].source={digest}",
            f"mapping['a'].source={digest}",
        ],
    )
    source.manifest_path.write_text(source.manifest_path.read_text() + "\n")
    with pytest.raises(HashMismatchError):
        await verbs.run_verb(fake.SPEC, cfg, out=tmp_path / "out")


async def test_nested_verb_warnings_reach_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(warnings, "showwarning", verbs._showwarning)
    original = fake.echo

    async def nested(cfg: fake.EchoConfig, run: RunDir) -> Handle:
        if cfg.message == "parent":
            await verbs.run_verb(fake.SPEC, fake.EchoConfig(message="child"), out=run.path("child"))
        else:
            warnings.warn("child degraded", stacklevel=1)
        return await original(cfg, run)

    monkeypatch.setattr(fake, "echo", nested)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        result = await verbs.run_verb(
            fake.SPEC, fake.EchoConfig(message="parent"), out=tmp_path / "out"
        )
    assert result.warnings == ["child degraded"]


async def test_resume_hook_never_bypasses_hash_and_recovers_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marli.errors import HashMismatchError

    cfg = fake.EchoConfig()
    await verbs.run_verb(fake.SPEC, cfg, out=tmp_path)
    calls = []

    def should_resume(handle: Handle, cfg: fake.EchoConfig) -> bool:
        calls.append(handle)
        return True

    monkeypatch.setattr(fake, "should_resume", should_resume, raising=False)
    with pytest.raises(HashMismatchError):
        await verbs.run_verb(fake.SPEC, dataclasses.replace(cfg, message="changed"), out=tmp_path)
    assert not calls
    original = fake.echo

    async def crash(cfg: fake.EchoConfig, run: RunDir) -> Handle:
        assert run.status is RunStatus.RESUME
        assert not run.path(fake.SPEC.manifest).exists()
        raise RuntimeError("interrupted reopened run")

    monkeypatch.setattr(fake, "echo", crash)
    with pytest.raises(RuntimeError, match="interrupted reopened"):
        await verbs.run_verb(fake.SPEC, cfg, out=tmp_path)
    monkeypatch.setattr(fake, "echo", original)
    result = await verbs.run_verb(fake.SPEC, cfg, out=tmp_path)
    assert result.status is RunStatus.RESUME
    assert len(result.handle.file("rows").read_text().splitlines()) == cfg.n
