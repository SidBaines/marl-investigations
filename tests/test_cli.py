"""The command shell stays machine-readable through reuse, recovery, and errors."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path

import _marli_fake_verbs as fake
import pytest
import yaml

from marli import __version__, registry, verbs
from marli.cli.main import main
from marli.errors import BackendError, BudgetExceededError
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


@pytest.mark.parametrize("runs_root", [None, "custom-runs"])
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
    assert markdown.startswith("<!-- Generated:")
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
    "error", [RuntimeError("oops"), BudgetExceededError("budget"), BackendError("backend")]
)
def test_execution_errors_are_json(tmp_path: Path, monkeypatch, capsys, error: Exception) -> None:
    async def fail(cfg, run):
        print("junk before failure")
        raise error

    monkeypatch.setattr(fake, "echo", fail)
    code = getattr(error, "exit_code", 1)
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
    assert ("Traceback" in captured.err) is (code == 1)


def test_help_is_lazy(monkeypatch, capsys) -> None:
    spec = replace(fake.SPEC, fn="unimportable_backend:fn", config="unimportable_backend:Config")
    monkeypatch.setattr(verbs, "VERBS", {spec.name: spec})
    for args in (["--help"], ["debug", "--help"], ["debug", "echo", "--help"]):
        with pytest.raises(SystemExit) as caught:
            main(args)
        assert caught.value.code == 0
        assert "usage: marli" in capsys.readouterr().out
