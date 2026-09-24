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
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from marli import config, handles
from marli.errors import ConfigError, MarliError
from marli.handles import Handle
from marli.rundir import RunDir, RunStatus

_warning_collector: ContextVar[list[str] | None] = ContextVar("marli_warnings", default=None)
_original_showwarning = warnings.showwarning


def _showwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    collector = _warning_collector.get()
    if collector is None:
        _original_showwarning(message, category, filename, lineno, file, line)
    else:
        collector.append(str(message))


# Install once: task-local collectors must not replace each other's warning hooks.
warnings.showwarning = _showwarning


@dataclass(frozen=True)
class VerbSpec:
    """Lazy targets for an async ``fn(cfg, run: RunDir) -> Handle`` verb.

    ``config`` names its dataclass; the returned handle must use ``manifest``.
    """

    name: str
    fn: str
    config: str
    manifest: str
    help: str


VERBS: dict[str, VerbSpec] = {
    "serve vllm": VerbSpec(
        "serve vllm",
        "marli.serve.vllm:vllm",
        "marli.serve.vllm:VLLMServeConfig",
        "server.json",
        "Supervise a token-native vLLM server with versioned runtime LoRA loading.",
    ),
    "serve status": VerbSpec(
        "serve status",
        "marli.serve.vllm:status",
        "marli.serve.vllm:ServerControlConfig",
        "server-status.json",
        "Record server process liveness and HTTP readiness.",
    ),
    "serve stop": VerbSpec(
        "serve stop",
        "marli.serve.vllm:stop",
        "marli.serve.vllm:ServerControlConfig",
        "server-status.json",
        "Stop the server process group, escalating after the grace period.",
    ),
    "data sft": VerbSpec(
        "data sft",
        "marli.data.sft:data_sft",
        "marli.data.sft:DataSFTConfig",
        "sft.json",
        "Build exact-token SFT datums from rejection-filtered teacher episodes.",
    ),
    "train sft": VerbSpec(
        "train sft",
        "marli.train.sft:train_sft",
        "marli.train.sft:TrainSFTConfig",
        "checkpoint.json",
        "Warm-start a student with cross-entropy and resumable token batches.",
    ),
    "train rl": VerbSpec(
        "train rl",
        "marli.train.loop:train_rl",
        "marli.train.rl:TrainRLConfig",
        "checkpoint.json",
        "Train synchronous on-policy learners with checkpoint and optimizer resume.",
    ),
    "data build": VerbSpec(
        "data build",
        "marli.data.build:build",
        "marli.data.build:BuildConfig",
        "taskset.json",
        "Build a taskset from a source, optionally excluding overlapping prompts.",
    ),
    "data filter": VerbSpec(
        "data filter",
        "marli.data.filter:filter",
        "marli.data.filter:FilterConfig",
        "taskset.json",
        "Filter a taskset by pass rates from saved rollouts.",
    ),
    "eval rollout": VerbSpec(
        "eval rollout",
        "marli.eval.rollout:rollout",
        "marli.eval.rollout:RolloutConfig",
        "episodes.json",
        "Sample resumable episodes with bounded concurrency and a spend guard.",
    ),
    "eval score": VerbSpec(
        "eval score",
        "marli.eval.score:score",
        "marli.eval.score:ScoreConfig",
        "scores.json",
        "Score saved episodes; optionally regrade with the environment verifier.",
    ),
    "eval report": VerbSpec(
        "eval report",
        "marli.eval.report:report",
        "marli.eval.report:ReportConfig",
        "report.json",
        "Report Scores inputs with compute and paired task statistics.",
    ),
    "eval grid": VerbSpec(
        "eval grid",
        "marli.eval.grid:grid",
        "marli.eval.grid:GridConfig",
        "report.json",
        "Run labelled rollout/score cells and combine their compute-aware report.",
    ),
    "view": VerbSpec(
        "view",
        "marli.viewer.verb:view",
        "marli.viewer.verb:ViewConfig",
        "view.json",
        "Render saved multi-agent episodes as one self-contained HTML page.",
    ),
}
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
    """A durable handle and the run's status when opened, before execution.

    FRESH starts new work, RESUME continues incomplete work, and COMPLETE
    reuses a saved handle without calling the verb.
    """

    handle: Handle
    status: RunStatus
    manifest: Path
    config_hash: str
    warnings: list[str]


def input_paths(value: str) -> dict[str, str]:
    """Parse one manifest or label=path entries separated by os.pathsep."""
    if "=" not in value or Path(value).exists():
        return {"": value}
    entries: dict[str, str] = {}
    for entry in value.split(os.pathsep):
        label, sep, path = entry.partition("=")
        if not sep or not label or not path or label in entries:
            raise ConfigError("inputs require unique label=path entries joined by os.pathsep")
        entries[label] = path
    return entries


def resolve_inputs(value: Any, name: str = "") -> tuple[Any, list[str]]:
    """Copy nested configuration and hash each input's manifest at its field location.

    Runtime subtrees belong to their owning child run, so their inputs do not
    enter the parent's identity. Dict keys and list indices preserve placement.
    """
    digests: list[str] = []
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {}
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            key = f"{name}.{field.name}" if name else field.name
            if field.metadata.get("runtime"):
                continue
            if field.metadata.get("input") and item is not None:
                if isinstance(item, (str, Path)):
                    paths = input_paths(str(item)) if isinstance(item, str) else {"": item}
                    resolved = {}
                    for label, path in paths.items():
                        manifest = input_manifest(path, key)
                        location = f"{key}[{label!r}]" if label else key
                        digests.append(f"{location}={handles.sha256_file(manifest)}")
                        resolved[label] = str(manifest)
                    if isinstance(item, Path):
                        changes[field.name] = Path(resolved[""])
                    elif "" in resolved:
                        changes[field.name] = resolved[""]
                    else:
                        changes[field.name] = os.pathsep.join(
                            f"{label}={path}" for label, path in resolved.items()
                        )
                else:
                    raise ConfigError(f"input field {key!r} requires a manifest path")
            else:
                changes[field.name], nested = resolve_inputs(item, key)
                digests.extend(nested)
        return dataclasses.replace(value, **changes), digests
    if isinstance(value, (list, tuple, dict)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        copied = {}
        for key, item in items:
            copied[key], nested = resolve_inputs(item, f"{name}[{key!r}]")
            digests.extend(nested)
        if isinstance(value, dict):
            return copied, digests
        return type(value)(copied.values()), digests
    return value, digests


def input_manifest(value: str | Path, name: str) -> Path:
    """Resolve a file or a directory's single readable handle manifest."""
    path = Path(value).resolve()
    if path.is_dir():
        candidates = []
        for candidate in sorted(path.glob("*.json")):
            try:
                if not candidate.is_file() or candidate.stat().st_size > 1_000_000:
                    continue
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
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
    """Resolve inputs and own the run until its completion manifest is durable.

    A verb module may define ``should_resume(handle, cfg) -> bool`` to reopen a
    complete run, after RunDir has checked the unchanged configuration hash.
    """
    spec = get_verb(verb) if isinstance(verb, str) else verb
    fn, cls = resolve(spec)
    if not isinstance(cfg, cls):
        raise TypeError(f"{spec.name} requires {cls.__name__}, got {type(cfg).__name__}")
    cfg, digests = resolve_inputs(cfg)
    h = config.config_hash(cfg, input_digests=digests)
    if out == "auto":
        out = Path(os.environ.get("MARLI_RUNS") or "runs") / spec.name.replace(" ", "-") / h[:12]
    out = Path(out).resolve()
    if spec.name == "eval grid" and force:
        from marli.eval.grid import force_changed_cells

        cfg = force_changed_cells(cfg, out)
        # A grid owns shared cell storage: --force never wipes its directory.
        force = False
    caught: list[str] = []
    parent_warnings = _warning_collector.get()
    token = _warning_collector.set(caught)
    try:
        with RunDir(
            out, kind=spec.name, manifest_name=spec.manifest, config_hash=h, force=force
        ) as run:
            if run.status is RunStatus.COMPLETE:
                handle = handles.load_any(out / spec.manifest)
                should_resume = getattr(
                    importlib.import_module(spec.fn.split(":")[0]), "should_resume", None
                )
                if should_resume is not None and should_resume(handle, cfg):
                    run.reopen()
            opening_status = run.status
            if opening_status is not RunStatus.COMPLETE:
                config.save(cfg, out / "config.yaml")
                handle = await fn(cfg, run)
                returned_manifest = getattr(handle, "MANIFEST", None)
                if not isinstance(handle, Handle) or returned_manifest != spec.manifest:
                    raise MarliError(
                        f"verb {spec.name!r} expected a Handle with manifest {spec.manifest!r}; "
                        f"returned {type(handle).__name__} "
                        f"with manifest {returned_manifest!r}"
                    )
                handle = run.finalize(handle)
            caught.extend(handle.meta.get("warnings", []))
    finally:
        _warning_collector.reset(token)
        caught = list(dict.fromkeys(caught))
        if parent_warnings is not None:
            parent_warnings.extend(caught)
        else:
            for warning in caught:
                logging.getLogger("marli").warning("%s", warning)
    return VerbResult(handle, opening_status, handle.manifest_path, h, caught)
