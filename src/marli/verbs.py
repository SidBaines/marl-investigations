"""Lazy verb registration and one lifecycle for CLI and Python callers.

Resolving inputs before hashing makes run identity depend on manifest bytes;
RunDir then provides the same idempotency and recovery for every caller.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import logging
import os
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_type_hints

from marli import config, handles
from marli.errors import ConfigError
from marli.handles import Handle
from marli.rundir import RunDir, RunStatus


@dataclass(frozen=True)
class VerbSpec:
    name: str
    fn: str
    config: str
    manifest: str
    help: str


VERBS: dict[str, VerbSpec] = {}
BUILTINS: tuple[str, ...] = ("list", "describe", "inspect", "status")


def get_verb(name: str) -> VerbSpec:
    """Look up metadata without importing the implementation or its dependencies."""
    if name not in VERBS:
        raise ConfigError(f"unknown verb {name!r}; known verbs: {sorted(VERBS)}")
    return VERBS[name]


def _import_target(ref: str) -> Any:
    parts = ref.split(":")
    if (
        len(parts) != 2
        or not all(part.isidentifier() for part in parts[0].split("."))
        or not parts[1].isidentifier()
    ):
        raise ConfigError(f"invalid target {ref!r}; expected module:attr")
    try:
        return getattr(importlib.import_module(parts[0]), parts[1])
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot resolve target {ref!r}: {exc}") from exc


def resolve(spec: VerbSpec) -> tuple[Callable[..., Awaitable[Handle]], type]:
    """Import only the selected verb; listing commands must stay light."""
    fn = _import_target(spec.fn)
    cls = _import_target(spec.config)
    if not inspect.iscoroutinefunction(fn):
        raise ConfigError(f"verb target {spec.fn!r} must be a coroutine function")
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise ConfigError(f"config target {spec.config!r} must be a dataclass type")
    return fn, cls


@dataclass(frozen=True)
class VerbResult:
    handle: Handle
    status: RunStatus
    manifest: Path
    config_hash: str
    warnings: list[str]


def _check_input_fields(cls: type, prefix: str = "", ancestors: tuple[type, ...] = ()) -> None:
    if cls in ancestors:
        return
    hints = get_type_hints(cls)
    for field in dataclasses.fields(cls):
        name = f"{prefix}{field.name}"
        if prefix and field.metadata.get("input"):
            raise ConfigError(
                f"nested input field {name!r} is not supported; use a top-level field"
            )
        pending = [hints[field.name]]
        while pending:
            annotation = pending.pop()
            if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
                _check_input_fields(annotation, f"{name}.", (*ancestors, cls))
            else:
                pending.extend(get_args(annotation))


def _input_manifest(value: str | Path, name: str) -> Path:
    path = Path(value).resolve()
    if path.is_dir():
        candidates = []
        for candidate in sorted(path.glob("*.json")):
            if not candidate.is_file():
                continue
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and {"kind", "manifest_version"} <= data.keys():
                candidates.append(candidate)
        if len(candidates) != 1:
            raise ConfigError(
                f"input field {name!r}: expected exactly one handle manifest in {path}, "
                f"found {len(candidates)}"
            )
        path = candidates[0].resolve()
    if not path.is_file():
        raise ConfigError(f"input field {name!r}: manifest not found: {path}")
    return path


async def run_verb(
    verb: str | VerbSpec, cfg: Any, *, out: str | Path, force: bool = False
) -> VerbResult:
    """Resolve inputs and own the run until its completion manifest is durable."""
    spec = get_verb(verb) if isinstance(verb, str) else verb
    fn, cls = resolve(spec)
    if not isinstance(cfg, cls):
        raise TypeError(f"{spec.name} requires {cls.__name__}, got {type(cfg).__name__}")
    _check_input_fields(type(cfg))
    replacements = {}
    digests = []
    for field in dataclasses.fields(cfg):
        value = getattr(cfg, field.name)
        if field.metadata.get("input") and value is not None:
            manifest = _input_manifest(value, field.name)
            replacements[field.name] = str(manifest)
            digests.append(f"{field.name}={handles.sha256_file(manifest)}")
    cfg = dataclasses.replace(cfg, **replacements)
    h = config.config_hash(cfg, input_digests=digests)
    if out == "auto":
        out = Path(os.environ.get("MARLI_RUNS", "runs")) / spec.name.replace(" ", "-") / h[:12]
    out = Path(out).resolve()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with RunDir(
                out, kind=spec.name, manifest_name=spec.manifest, config_hash=h, force=force
            ) as run:
                opening_status = run.status
                if opening_status is RunStatus.COMPLETE:
                    handle = handles.load_any(out / spec.manifest)
                else:
                    config.save(cfg, out / "config.yaml")
                    handle = await fn(cfg, run)
                    handle = run.finalize(handle)
        finally:
            for warning in caught:
                logging.getLogger("marli").warning("%s", warning.message)
    return VerbResult(
        handle, opening_status, handle.manifest_path, h, [str(w.message) for w in caught]
    )
