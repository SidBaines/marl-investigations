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
)


def _violations(source: str, rule: str) -> list[int]:
    tree = ast.parse(source)
    asyncio_names = {"asyncio"}
    run_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            asyncio_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "asyncio"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "asyncio":
            run_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "run"
            )

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
            fn = node.func
            if rule == "asyncio.run":
                if (
                    isinstance(fn, ast.Attribute)
                    and fn.attr == "run"
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id in asyncio_names
                ) or (isinstance(fn, ast.Name) and fn.id in run_names):
                    found.append(node.lineno)
            elif (
                rule == "print"
                and isinstance(fn, ast.Name)
                and fn.id == "print"
                and not any(keyword.arg == "file" for keyword in node.keywords)
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
        ("print('junk')", "print", [1]),
        ("print('log', file=sys.stderr)", "print", []),
        ("runner.run()", "asyncio.run", []),
    ],
)
def test_boundary_scanner(source: str, rule: str, expected: list[int]) -> None:
    assert _violations(source, rule) == expected


def _python(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
    )


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
    for name, spec in verbs.VERBS.items():
        assert name == spec.name
        assert re.fullmatch(r"[a-z0-9-]+(?: [a-z0-9-]+)?", name), name
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
