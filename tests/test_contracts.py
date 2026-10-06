"""Repository-wide boundaries keep new verbs composable and safe to import."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from marli import verbs
from marli.cli.main import _build_parser

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "src" / "marli"
HEAVY = (
    "torch",
    "tinker",
    "tinker_cookbook",
    "vllm",
    "transformers",
    "datasets",
    "peft",
    "math_verify",
    "wandb",
    "inspect_ai",
    "litellm",
)


def _violations(source: str, rule: str) -> list[int]:
    tree = ast.parse(source)
    aliases = {
        "asyncio": "asyncio",
        "print": "builtins.print",
        "builtins": "builtins",
        "sys": "sys",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                aliases[name] = alias.name if alias.asname else name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def qualified_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{qualified_name(node.value)}.{node.attr}"
        return None

    found = []
    for node in ast.walk(tree):
        if rule == "argparse":
            if (
                isinstance(node, ast.Import)
                and any(
                    alias.name == "argparse" or alias.name.startswith("argparse.")
                    for alias in node.names
                )
            ) or (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "argparse" or node.module.startswith("argparse."))
            ):
                found.append(node.lineno)
        elif isinstance(node, ast.Call):
            fn = qualified_name(node.func)
            if rule == "asyncio.run":
                if fn in {
                    "asyncio.run",
                    "asyncio.runners.run",
                    "asyncio.Runner",
                    "asyncio.runners.Runner",
                }:
                    found.append(node.lineno)
            elif rule == "print" and fn == "builtins.print":
                file = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "file"), None
                )
                if (
                    file is None
                    or qualified_name(file) in {"sys.stdout", "sys.__stdout__"}
                    or (isinstance(file, ast.Constant) and file.value is None)
                ):
                    found.append(node.lineno)
    return found


@pytest.mark.parametrize("rule", ["argparse", "asyncio.run", "print"])
def test_cli_only_boundaries(rule: str) -> None:
    violations = []
    for path in sorted(LIBRARY.rglob("*.py")):
        if path.is_relative_to(LIBRARY / "cli"):
            continue
        violations.extend(
            f"{path.relative_to(ROOT)}:{line}" for line in _violations(path.read_text(), rule)
        )
    assert not violations, f"{rule} belongs only in src/marli/cli/: {violations}"


@pytest.mark.parametrize(
    ("source", "rule", "expected"),
    [
        ("import argparse as parser", "argparse", [1]),
        ("from argparse import ArgumentParser", "argparse", [1]),
        ("import asyncio as aio\naio.run(main())", "asyncio.run", [2]),
        ("from asyncio import run\nrun(main())", "asyncio.run", [2]),
        ("from asyncio import run as start\nstart(main())", "asyncio.run", [2]),
        ("asyncio.runners.run(main())", "asyncio.run", [1]),
        ("asyncio.Runner()", "asyncio.run", [1]),
        ("from asyncio import Runner as Start\nStart()", "asyncio.run", [2]),
        ("from asyncio import runners as r\nr.run(main())", "asyncio.run", [2]),
        ("import asyncio.runners as r\nr.Runner()", "asyncio.run", [2]),
        ("from asyncio.runners import run as start\nstart(main())", "asyncio.run", [2]),
        ("print('junk')", "print", [1]),
        ("print('junk', file=sys.stdout)", "print", [1]),
        ("print('junk', file=sys.__stdout__)", "print", [1]),
        ("print('junk', file=None)", "print", [1]),
        ("builtins.print('junk')", "print", [1]),
        ("builtins.print('junk', file=sys.stdout)", "print", [1]),
        ("import builtins as b\nb.print('junk')", "print", [2]),
        ("import sys as s\nprint('junk', file=s.__stdout__)", "print", [2]),
        ("from sys import stdout as out\nprint('junk', file=out)", "print", [2]),
        ("print('log', file=sys.stderr)", "print", []),
        ("builtins.print('log', file=sys.__stderr__)", "print", []),
        ("print('saved', file=stream)", "print", []),
        ("runner.run()", "asyncio.run", []),
    ],
)
def test_boundary_scanner(source: str, rule: str, expected: list[int]) -> None:
    assert _violations(source, rule) == expected


def _python(*args: str) -> subprocess.CompletedProcess[str]:
    pythonpath = os.pathsep.join(filter(None, (str(ROOT / "src"), os.environ.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_python_helper_reports_child_stderr() -> None:
    with pytest.raises(AssertionError, match="child failure detail"):
        _python("-c", "import sys; sys.stderr.write('child failure detail'); sys.exit(9)")


def test_python_helper_preserves_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = os.pathsep.join((str(ROOT / "tests"), str(ROOT / "examples")))
    monkeypatch.setenv("PYTHONPATH", existing)
    result = _python("-c", "import os; print(os.environ['PYTHONPATH'])")
    assert result.stdout.strip() == str(ROOT / "src") + os.pathsep + existing


def test_warning_hook_is_installed_without_changing_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, (str(ROOT / "tests"), os.environ.get("PYTHONPATH")))),
    )
    result = _python(
        "-c",
        f"""
import asyncio
import warnings
before = warnings.filters[:]
from marli import verbs
import _marli_fake_verbs as fake
echo = fake.echo
async def warn(cfg, run):
    warnings.warn("captured warning")
    return await echo(cfg, run)
fake.echo = warn
result = asyncio.run(verbs.run_verb(fake.SPEC, fake.EchoConfig(), out={str(tmp_path)!r}))
assert result.warnings == ["captured warning"], result.warnings
assert warnings.filters == before
warnings.warn("outside warning")
""",
    )
    assert "UserWarning: outside warning" in result.stderr


@pytest.mark.parametrize("help_requested", [False, True])
def test_imports_stay_light(help_requested: bool) -> None:
    modules = (
        "marli",
        "marli.cli.main",
        "marli.verbs",
        "marli.config",
        "marli.registry",
        "marli.handles",
        "marli.rundir",
        "marli.runlog",
        "marli.errors",
    )
    _python(
        "-c",
        f"""
import importlib
import sys
for module in {modules!r}:
    importlib.import_module(module)
if {help_requested!r}:
    from marli.cli.main import main
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
loaded = sorted(set({HEAVY!r}).intersection(sys.modules))
assert not loaded, loaded
""",
    )


def test_cli_module_help() -> None:
    result = _python("-m", "marli.cli.main", "--help")
    assert result.returncode == 0
    assert "usage: marli" in result.stdout


def test_registered_verb_contracts() -> None:
    _build_parser()
    for name, spec in verbs.VERBS.items():
        assert name == spec.name
        assert re.fullmatch(r"[a-z0-9][a-z0-9-]*( [a-z0-9][a-z0-9-]*)?", name), name
        assert spec.manifest.endswith(".json"), name
        fn, cls = verbs.resolve(spec)
        assert inspect.iscoroutinefunction(fn), name
        parameters = inspect.signature(fn).parameters
        assert list(parameters) == ["cfg", "run"], name
        assert all(
            parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is inspect.Parameter.empty
            for parameter in parameters.values()
        ), name
        assert isinstance(cls, type) and dataclasses.is_dataclass(cls), name
        assert not {field.name for field in dataclasses.fields(cls)} & {"out", "force"}, name


def test_generated_cli_reference_is_current() -> None:
    result = _python("-m", "marli.cli.main", "describe", "--all", "--markdown")
    assert (ROOT / "docs" / "cli.md").read_text(encoding="utf-8") == result.stdout, (
        "Regenerate with: uv run marli describe --all --markdown > docs/cli.md"
    )
