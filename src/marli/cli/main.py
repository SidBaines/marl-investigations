"""Thin command shell: agents chain on exactly one JSON result line.

Verb targets stay as strings until execution or description, so help never
imports backend stacks. The shared runner owns all run lifecycle decisions.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

from marli import (
    __version__,
    config,
    handles,
    registry,
    rundir,  # noqa: F401 -- include run-directory errors in the reference.
    verbs,
)
from marli.errors import ConfigError, MarliError


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ConfigError(message)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="marli", description="Multi-agent RL investigations")
    parser.add_argument("--version", action="store_true", help="show the package version")
    commands = parser.add_subparsers(dest="command")
    listing = commands.add_parser("list", help="list file-backed registry entries")
    listing.add_argument("kind", nargs="?")
    describe = commands.add_parser("describe", help="describe verb configuration")
    describe.add_argument("verb", nargs="*")
    describe.add_argument("--all", action="store_true", help="describe every registered verb")
    describe.add_argument("--markdown", action="store_true", help="emit a Markdown reference")
    inspect = commands.add_parser("inspect", help="inspect a manifest's provenance chain")
    inspect.add_argument("path")
    status = commands.add_parser("status", help="read run status without taking its lock")
    status.add_argument("out")

    groups = {}
    single_names = {spec.name for spec in verbs.VERBS.values() if " " not in spec.name}
    for name, spec in sorted(verbs.VERBS.items()):
        if (
            name != spec.name
            or re.fullmatch(r"[a-z0-9][a-z0-9-]*( [a-z0-9][a-z0-9-]*)?", name) is None
        ):
            raise ConfigError(f"invalid verb table entry {name!r}: expected 1–2 lowercase words")
        words = name.split(" ")
        if words[0] in verbs.BUILTINS:
            raise ConfigError(f"verb {name!r} collides with builtin {words[0]!r}")
        if len(words) == 2:
            group, word = words
            if group in single_names:
                raise ConfigError(f"verb group {group!r} collides with a one-word verb")
            if group not in groups:
                help_text = f"{group} verbs"
                groups[group] = commands.add_parser(
                    group, help=help_text, description=help_text
                ).add_subparsers(required=True)
            command = groups[group].add_parser(word, help=spec.help, description=spec.help)
        else:
            command = commands.add_parser(name, help=spec.help, description=spec.help)
        command.set_defaults(verb_name=name)
        command.add_argument("config_args", nargs="*", metavar="CONFIG.yaml|key=value")
        command.add_argument("--out", required=True, help="output directory or auto")
        command.add_argument("--force", action="store_true", help="replace existing run output")
    return parser


def _read_optional_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _status(out: str) -> dict[str, Any]:
    path = Path(out).resolve()
    run = _read_optional_json(path / ".marli" / "run.json")
    progress = _read_optional_json(path / "progress.json")
    manifest = path / run["manifest"] if run is not None else None
    complete = manifest is not None and manifest.is_file()
    return {
        "ok": True,
        "kind": "status",
        "out": str(path),
        "run": run,
        "progress": progress,
        "manifest": str(manifest) if complete else None,
        "complete": complete,
    }


def _markdown_cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return text.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def _error_types(cls: type[MarliError]) -> Iterator[type[MarliError]]:
    yield cls
    for child in cls.__subclasses__():
        yield from _error_types(child)


def _markdown(descriptions: list[dict[str, Any]]) -> str:
    error_groups: dict[int, set[type[MarliError]]] = {}
    for cls in _error_types(MarliError):
        error_groups.setdefault(cls.exit_code, set()).add(cls)
    error_rows = []
    for code, classes in sorted(error_groups.items()):
        meaning = "; ".join(
            f"`{cls.__name__}`: {cls.__doc__ or ''}"
            for cls in sorted(classes, key=lambda c: c.__name__)
        )
        error_rows.append(f"| {code} | {_markdown_cell(meaning)} |")
    lines = [
        "Generated: `uv run marli describe --all --markdown > docs/cli.md`",
        "",
        "# marli CLI reference",
        "",
        "## Built-in commands",
        "",
        "| Command | Purpose |",
        "| --- | --- |",
        "| `marli --version` | Report the package version. |",
        "| `marli list [KIND]` | List all registries, or the entries in one registry. |",
        "| `marli describe VERB... [--markdown]` | Describe one verb and its config fields. |",
        "| `marli describe --all [--markdown]` | Describe every registered verb. |",
        "| `marli inspect PATH` | Inspect a manifest file or directory and its input chain. |",
        "| `marli status OUT` | Read saved run and progress records without taking a lock. |",
        "",
        "## Output and exit codes",
        "",
        "Commands emit exactly one compact JSON line on stdout; logs and stray verb",
        "prints (including subprocess output) go to stderr. Success includes `ok: true`;",
        "errors include `ok: false`,",
        "`error`, `message`, and `exit_code`. `describe --markdown` emits Markdown instead",
        "of JSON. `--help` prints normal argparse help to stdout and exits with code 0.",
        "A verb's `SystemExit` is an unexpected failure (exit 1); `KeyboardInterrupt` exits 130.",
        "",
        "| Exit code | Meaning |",
        "| --- | --- |",
        "| 0 | Success |",
        *error_rows,
        "| 130 | Interrupted (`KeyboardInterrupt`) |",
        "",
        "## Verbs",
        "",
        "Usage: `marli VERB... [CONFIG.yaml ...] [key=value ...] --out DIR|auto [--force]`.",
        "YAML files merge left to right, followed by dotted overrides. `--out auto` uses",
        "`$MARLI_RUNS/<verb-with-hyphens>/<hash12>` (default root: `runs`). Matching completed",
        "runs are reused; incomplete runs resume. `--force` replaces existing run output.",
    ]
    if not descriptions:
        lines.extend(["", "No verbs registered yet."])
    for description in descriptions:
        lines.extend(
            [
                "",
                f"### {description['name']}",
                "",
                description["help"],
                "",
                f"Manifest produced: `{description['manifest']}`.",
                "",
                "| Name | Type | Default | Required | Runtime | Input | Help |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for field in description["config"]:
            values = [
                field[key] for key in ("name", "type", "default", "required", "runtime", "input")
            ]
            values.append(field["help"] or "")
            lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def _dispatch(args: argparse.Namespace) -> dict[str, Any] | str:
    if args.version:
        if args.command is not None:
            raise ConfigError("--version cannot be combined with a command")
        return {"ok": True, "kind": "version", "version": __version__}
    if args.command == "list":
        if args.kind is not None:
            return {
                "ok": True,
                "kind": "list",
                "registry": args.kind,
                "names": registry.catalog_names(args.kind),
            }
        return {
            "ok": True,
            "kind": "list",
            "registries": {kind: registry.catalog_names(kind) for kind in registry.catalog_kinds()},
        }
    if args.command == "describe":
        if bool(args.verb) == args.all:
            raise ConfigError("describe requires either a verb name or --all")
        specs = (
            [verbs.get_verb(name) for name in sorted(verbs.VERBS)]
            if args.all
            else [verbs.get_verb(" ".join(args.verb))]
        )
        descriptions = [
            {
                "name": spec.name,
                "help": spec.help,
                "manifest": spec.manifest,
                "config": config.describe_config(verbs.resolve(spec)[1]),
            }
            for spec in specs
        ]
        if args.markdown:
            return _markdown(descriptions)
        return {"ok": True, "kind": "describe", "verbs": descriptions}
    if args.command == "inspect":
        path = verbs.input_manifest(args.path, "PATH")
        return {"ok": True, "kind": "inspect", "chain": handles.inspect_chain(path)}
    if args.command == "status":
        return _status(args.out)
    if args.command is None:
        raise ConfigError("a command is required")
    spec = verbs.get_verb(args.verb_name)
    _, cls = verbs.resolve(spec)
    for y in args.config_args:
        if "=" not in y and not Path(y).is_file():
            raise ConfigError(f"config file not found: {y} (overrides must be key=value)")
    cfg = config.parse(cls, args.config_args)
    result = asyncio.run(verbs.run_verb(spec, cfg, out=args.out, force=args.force))
    summary = result.handle.summary()
    reserved = {
        "ok",
        "verb",
        "kind",
        "manifest",
        "status",
        "config_hash",
        "warnings",
        "error",
        "exit_code",
    }
    if overlap := summary.keys() & reserved:
        raise MarliError(f"verb {spec.name!r} summary overwrites contract keys: {sorted(overlap)}")
    return {
        "ok": True,
        "verb": spec.name,
        "kind": result.handle.KIND,
        "manifest": str(result.manifest),
        "status": result.status.value,
        "config_hash": result.config_hash,
        "warnings": result.warnings,
        **summary,
    }


def _error_output(exc: BaseException) -> tuple[int, str]:
    if isinstance(exc, MarliError):
        code = exc.exit_code
    elif isinstance(exc, KeyboardInterrupt):
        code = 130
    else:
        code = 1
    if not isinstance(exc, MarliError):
        traceback.print_exc(file=sys.stderr)
    return code, json.dumps(
        {"ok": False, "error": type(exc).__name__, "message": str(exc), "exit_code": code},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Translate every failure to JSON so callers never have to parse stderr."""
    exit_code = 0
    try:
        args, leftover = _build_parser().parse_known_args(argv)
        if leftover:
            if not hasattr(args, "verb_name") or any(arg.startswith("-") for arg in leftover):
                raise ConfigError(f"unrecognized arguments: {' '.join(leftover)}")
            args.config_args.extend(leftover)
    except Exception as exc:
        exit_code, output = _error_output(exc)
    else:
        try:
            level = os.environ.get("MARLI_LOG_LEVEL", "INFO").upper()
            if level not in logging.getLevelNamesMapping():
                raise ConfigError(f"invalid MARLI_LOG_LEVEL: {level!r}")
            logging.basicConfig(stream=sys.stderr, level=level)
            sys.stdout.flush()
            saved = os.dup(1)
            try:
                os.dup2(2, 1)
                with contextlib.redirect_stdout(sys.stderr):
                    result = _dispatch(args)
                    output = (
                        result
                        if isinstance(result, str)
                        else json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
                    )
            finally:
                try:
                    sys.stdout.flush()
                finally:
                    os.dup2(saved, 1)
                    os.close(saved)
        except (Exception, SystemExit, KeyboardInterrupt) as exc:
            exit_code, output = _error_output(exc)
    sys.stdout.write(output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
