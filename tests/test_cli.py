"""The command shell stays machine-readable through reuse, recovery, and errors."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import _marli_fake_verbs as fake
import pytest
import yaml

from marli import __version__, registry, verbs
from marli.cli.main import main
from marli.errors import BackendError, BudgetExceededError, ConfigError
from marli.handles import Handle
from marli.registry import Registry
from marli.rundir import RunDir


@pytest.fixture(autouse=True)
def echo_verb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verbs, "VERBS", {fake.SPEC.name: fake.SPEC})
    monkeypatch.setattr(fake, "invocations", 0)
    monkeypatch.setattr(fake, "crash_after", None)


def invoke(capsys: pytest.CaptureFixture[str], *args: str, code: int = 0) -> dict:
    assert main(list(args)) == code
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out)
    assert captured.out == json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    assert payload["ok"] is (code == 0)
    return payload


def rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]


def test_fresh_reuse_runtime_mismatch_and_force(tmp_path: Path, capsys) -> None:
    args = ["debug", "echo", "--out", str(tmp_path)]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    result = json.loads(captured.out)
    assert result == {
        "ok": True,
        "verb": "debug echo",
        "kind": "dummy",
        "manifest": str(tmp_path / "dummy.json"),
        "status": "fresh",
        "config_hash": verbs.config.config_hash(fake.EchoConfig()),
        "warnings": [],
        "n": 3,
    }
    assert "junk from echo" in captured.err
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "dummy.json").is_file()
    assert fake.invocations == 1
    assert invoke(capsys, *args)["status"] == "complete"
    assert (
        invoke(capsys, "debug", "echo", "concurrency=8", "--out", str(tmp_path))["status"]
        == "complete"
    )
    assert fake.invocations == 1
    mismatch = invoke(capsys, "debug", "echo", "n=5", "--out", str(tmp_path), code=3)
    assert mismatch["error"] == "HashMismatchError"
    assert mismatch["exit_code"] == 3
    assert len(rows(tmp_path)) == 3
    forced = invoke(capsys, "debug", "echo", "n=5", "--out", str(tmp_path), "--force")
    assert forced["status"] == "fresh"
    assert forced["n"] == 5
    assert rows(tmp_path) == [{"i": i, "message": "hi"} for i in range(5)]
    assert fake.invocations == 2


def test_resume_and_crashed_status(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(fake, "crash_after", 2)
    with pytest.raises(RuntimeError, match="simulated crash"):
        asyncio.run(verbs.run_verb(fake.SPEC, fake.EchoConfig(), out=tmp_path))
    capsys.readouterr()
    assert len(rows(tmp_path)) == 2
    status = invoke(capsys, "status", str(tmp_path))
    assert status["complete"] is False
    assert status["manifest"] is None
    assert status["run"]["kind"] == "debug echo"
    assert status["progress"]["done"] == 2
    monkeypatch.setattr(fake, "crash_after", None)
    result = invoke(capsys, "debug", "echo", "--out", str(tmp_path))
    assert result["status"] == "resume"
    assert fake.invocations == 2
    assert rows(tmp_path) == [{"i": i, "message": "hi"} for i in range(3)]


@pytest.mark.parametrize("runs_root", [None, "", "custom-runs"])
def test_auto_output(tmp_path: Path, monkeypatch, capsys, runs_root: str | None) -> None:
    monkeypatch.chdir(tmp_path)
    if runs_root is None:
        monkeypatch.delenv("MARLI_RUNS", raising=False)
    else:
        monkeypatch.setenv("MARLI_RUNS", runs_root)
    result = invoke(capsys, "debug", "echo", "--out", "auto")
    expected = tmp_path / (runs_root or "runs") / "debug-echo" / result["config_hash"][:12]
    assert result["manifest"] == str(expected / "dummy.json")
    assert expected.is_dir()


def test_input_resolution_and_digest(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    invoke(capsys, "debug", "echo", "--out", "source")
    args = ["debug", "echo", "source=source", "--out", "result"]
    result = invoke(capsys, *args)
    source_manifest = tmp_path / "source" / "dummy.json"
    saved = yaml.safe_load((tmp_path / "result" / "config.yaml").read_text())
    assert saved["source"] == str(source_manifest)
    chain = invoke(capsys, "inspect", result["manifest"])["chain"]
    assert chain["kind"] == "dummy"
    assert chain["inputs"][0]["path"] == str(source_manifest)
    assert chain["inputs"][0]["sha256_ok"] is True
    assert invoke(capsys, *args)["status"] == "complete"
    source_manifest.write_text(source_manifest.read_text() + "\n")
    assert invoke(capsys, *args, code=3)["error"] == "HashMismatchError"


def test_config_layers_and_one_word_verb(tmp_path: Path, monkeypatch, capsys) -> None:
    spec = replace(fake.SPEC, name="echo")
    monkeypatch.setattr(verbs, "VERBS", {spec.name: spec})
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("message: first\nn: 1\n")
    second.write_text("message: second\nn: 2\n")
    out = tmp_path / "out"
    result = invoke(capsys, "echo", str(first), str(second), "n=4", "--out", str(out))
    assert result["verb"] == "echo"
    assert rows(out) == [{"i": i, "message": "second"} for i in range(4)]


@pytest.mark.parametrize(
    "args",
    [
        ["debug", "echo", "unknown=5", "--out", "out"],
        ["debug", "echo"],
        ["unknown"],
        ["debug"],
        [],
        ["describe"],
        ["describe", "debug", "echo", "--all"],
        ["--version", "list"],
        ["--version", "debug", "echo", "--out", "out"],
        ["debug", "echo", "n=1", "--out", "out", "--unknown"],
        ["list", "extra", "leftover"],
        ["status", "out", "extra"],
    ],
)
def test_usage_errors(tmp_path: Path, monkeypatch, capsys, args: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    result = invoke(capsys, *args, code=2)
    assert result["error"] == "ConfigError"
    assert result["exit_code"] == 2
    assert not (tmp_path / "out").exists()


@dataclass
class RegistryEntry:
    name: str


def test_list(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "beta.yaml").write_text("{}\n")
    (tmp_path / "alpha.yaml").write_text("{}\n")
    monkeypatch.setattr(
        fake, "test_registry", Registry("test", tmp_path, RegistryEntry), raising=False
    )
    monkeypatch.setattr(registry, "CATALOG", {"test": "_marli_fake_verbs:test_registry"})
    assert invoke(capsys, "list") == {
        "ok": True,
        "kind": "list",
        "registries": {"test": ["alpha", "beta"]},
    }
    assert invoke(capsys, "list", "test") == {
        "ok": True,
        "kind": "list",
        "registry": "test",
        "names": ["alpha", "beta"],
    }
    assert invoke(capsys, "list", "unknown", code=2)["error"] == "ConfigError"


def test_describe_and_markdown(capsys, monkeypatch) -> None:
    result = invoke(capsys, "describe", "debug", "echo")
    assert result["kind"] == "describe"
    (description,) = result["verbs"]
    assert description["name"] == "debug echo"
    assert description["help"] == fake.SPEC.help
    assert description["manifest"] == "dummy.json"
    fields = {field["name"]: field for field in description["config"]}
    assert fields["concurrency"]["runtime"] is True
    assert fields["source"]["input"] is True
    assert fields["message"]["default"] == "hi"
    assert invoke(capsys, "describe", "--all") == result
    another = replace(fake.SPEC, name="alpha", help="Earlier verb.")
    monkeypatch.setitem(verbs.VERBS, another.name, another)
    assert main(["describe", "--all", "--markdown"]) == 0
    markdown = capsys.readouterr().out
    assert markdown.startswith("Generated: `")
    assert "`RunDirLockedError`" in markdown
    assert markdown.index("### alpha") < markdown.index("### debug echo")
    assert "| source | str \\| None | null | false | false | true |" in markdown
    assert "| concurrency | int | 4 | false | true | false |" in markdown


def test_status_complete_locked_and_missing(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "absent"
    assert invoke(capsys, "status", str(missing)) == {
        "ok": True,
        "kind": "status",
        "out": str(missing),
        "run": None,
        "progress": None,
        "manifest": None,
        "complete": False,
    }
    assert not missing.exists()
    result = invoke(capsys, "debug", "echo", "--out", str(tmp_path))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with RunDir(
        tmp_path, kind=fake.SPEC.name, manifest_name="dummy.json", config_hash=result["config_hash"]
    ):
        status = invoke(capsys, "status", str(tmp_path))
    assert status["complete"] is True
    assert status["manifest"] == result["manifest"]
    assert status["progress"]["done"] == 3
    assert before == {
        p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }


def test_version(capsys) -> None:
    assert invoke(capsys, "--version") == {
        "ok": True,
        "kind": "version",
        "version": __version__,
    }


@pytest.mark.parametrize("name", ["list foo", "list", "debug"])
def test_colliding_verb_names(monkeypatch, capsys, name: str) -> None:
    monkeypatch.setitem(verbs.VERBS, name, replace(fake.SPEC, name=name))
    result = invoke(capsys, "list", code=2)
    assert result["error"] == "ConfigError"
    assert "collides" in result["message"]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("oops"),
        BudgetExceededError("budget"),
        BackendError("backend"),
        SystemExit(0),
        SystemExit(7),
        KeyboardInterrupt(),
    ],
)
def test_execution_errors_are_json(
    tmp_path: Path, monkeypatch, capsys, error: BaseException
) -> None:
    async def fail(cfg, run):
        print("junk before failure")
        raise error

    monkeypatch.setattr(fake, "echo", fail)
    code = 130 if isinstance(error, KeyboardInterrupt) else getattr(error, "exit_code", 1)
    assert main(["debug", "echo", "--out", str(tmp_path)]) == code
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out) == {
        "ok": False,
        "error": type(error).__name__,
        "message": str(error),
        "exit_code": code,
    }
    assert "junk before failure" in captured.err
    assert ("Traceback" in captured.err) is (code in {1, 130})


def test_help_is_lazy(monkeypatch, capsys) -> None:
    spec = replace(fake.SPEC, fn="unimportable_backend:fn", config="unimportable_backend:Config")
    monkeypatch.setattr(verbs, "VERBS", {spec.name: spec})
    for args in (["--help"], ["debug", "--help"], ["debug", "echo", "--help"]):
        with pytest.raises(SystemExit) as caught:
            main(args)
        assert caught.value.code == 0
        assert "usage: marli" in capsys.readouterr().out


@pytest.mark.parametrize("error", [None, RuntimeError("oops"), SystemExit(9), KeyboardInterrupt()])
def test_fd_stdout_is_redirected_and_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    error: BaseException | None,
) -> None:
    echo = fake.echo

    async def noisy(cfg: fake.EchoConfig, run: RunDir) -> Handle:
        os.write(1, b"junk\n")
        subprocess.run(["echo", "x"], check=True)
        if error is not None:
            raise error
        return await echo(cfg, run)

    monkeypatch.setattr(fake, "echo", noisy)
    code = 0 if error is None else 130 if isinstance(error, KeyboardInterrupt) else 1
    assert main(["debug", "echo", "--out", str(tmp_path)]) == code
    captured = capfd.readouterr()
    assert len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out)
    assert payload["ok"] is (error is None)
    if error is not None:
        assert payload["error"] == type(error).__name__
        assert payload["exit_code"] == code
    assert "junk" in captured.err.splitlines()
    assert "x" in captured.err.splitlines()
    os.write(1, b"restored\n")
    captured = capfd.readouterr()
    assert captured.out == "restored\n"
    assert captured.err == ""


@pytest.mark.parametrize("path_kind", ["missing", "directory", "bare_override"])
def test_invalid_config_paths_are_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], path_kind: str
) -> None:
    path = str(tmp_path if path_kind == "directory" else tmp_path / "missing.yaml")
    if path_kind == "bare_override":
        path = "message"
    out = tmp_path / "out"
    result = invoke(capsys, "debug", "echo", path, "--out", str(out), code=2)
    assert result["error"] == "ConfigError"
    assert result["message"] == f"config file not found: {path} (overrides must be key=value)"
    assert not out.exists()


def test_inspect_resolves_directory_and_missing_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = invoke(capsys, "debug", "echo", "--out", str(tmp_path / "source"))
    expected = invoke(capsys, "inspect", source["manifest"])
    assert invoke(capsys, "inspect", str(tmp_path / "source")) == expected
    result = invoke(capsys, "inspect", str(tmp_path / "missing"), code=2)
    assert result["error"] == "ConfigError"
    assert "manifest not found" in result["message"]
    result = invoke(capsys, "inspect", str(tmp_path), code=2)
    assert "expected exactly one handle manifest" in result["message"]


def test_config_args_split_around_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("n: 1\nmessage: first\n")
    second.write_text("n: 2\nmessage: second\n")
    out = tmp_path / "out"
    result = invoke(
        capsys,
        "debug",
        "echo",
        str(first),
        "message=before",
        "--out",
        str(out),
        str(second),
        "message=after",
        "--force",
        "n=4",
    )
    assert result["n"] == 4
    assert rows(out) == [{"i": i, "message": "after"} for i in range(4)]


@pytest.mark.parametrize(
    "key",
    ["ok", "verb", "kind", "manifest", "status", "config_hash", "warnings", "error", "exit_code"],
)
def test_summary_cannot_overwrite_contract_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], key: str
) -> None:
    def summary(self: fake.DummyHandle) -> dict[str, Any]:
        return {key: "corrupted"}

    monkeypatch.setattr(fake.DummyHandle, "summary", summary)
    result = invoke(capsys, "debug", "echo", "--out", str(tmp_path), code=1)
    assert result["error"] == "MarliError"
    assert "summary overwrites contract keys" in result["message"]
    assert key in result["message"]


@pytest.mark.parametrize("level", ["debug", "iNfO", "WARNING", "error"])
def test_log_level_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], level: str
) -> None:
    configured: list[str] = []

    def configure(**kwargs: Any) -> None:
        configured.append(kwargs["level"])

    monkeypatch.setenv("MARLI_LOG_LEVEL", level)
    monkeypatch.setattr(logging, "basicConfig", configure)
    invoke(capsys, "--version")
    assert configured == [level.upper()]


@pytest.mark.parametrize("level", ["invalid", "", "20"])
def test_invalid_log_level_is_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], level: str
) -> None:
    monkeypatch.setenv("MARLI_LOG_LEVEL", level)
    out = tmp_path / "out"
    result = invoke(capsys, "debug", "echo", "--out", str(out), code=2)
    assert result["error"] == "ConfigError"
    assert "MARLI_LOG_LEVEL" in result["message"]
    assert not out.exists()


@pytest.mark.parametrize(
    "name", ["-echo", "debug -echo", "-debug echo", "debug echo extra", "Echo"]
)
def test_invalid_verb_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    monkeypatch.setattr(verbs, "VERBS", {name: replace(fake.SPEC, name=name)})
    result = invoke(capsys, "list", code=2)
    assert "invalid verb table entry" in result["message"]


def test_markdown_groups_error_subclasses_recursively(capsys: pytest.CaptureFixture[str]) -> None:
    class SpecificConfigError(ConfigError):
        """A specific configuration failure."""

    class NestedConfigError(SpecificConfigError):
        """A nested configuration failure."""

    assert main(["describe", "--all", "--markdown"]) == 0
    markdown = capsys.readouterr().out
    row = next(line for line in markdown.splitlines() if line.startswith("| 2 |"))
    for cls in (ConfigError, SpecificConfigError, NestedConfigError):
        assert f"`{cls.__name__}`" in row
    assert sum(line.startswith("| 2 |") for line in markdown.splitlines()) == 1


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--help"], ["debug verbs"]),
        (["debug", "--help"], ["debug verbs"]),
        (["describe", "--help"], ["describe every registered verb", "emit a Markdown reference"]),
    ],
)
def test_help_explains_options_and_groups(
    capsys: pytest.CaptureFixture[str], args: list[str], expected: list[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(args)
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    for text in expected:
        assert text in help_text
