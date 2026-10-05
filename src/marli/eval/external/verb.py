"""``marli eval external``: one external suite against a list of policies (cells).

- ``inspect`` suites: each cell names a policy (a vLLM ref to the base model or an
  adapter). The verb probes the server for that name, runs the Inspect task through
  Inspect's Python API with the resolved settings, keeps Inspect's ``.eval`` logs
  under ``logs/<cell>/`` and reads them into rows. A finished cell is skipped on
  resume; an interrupted one is rerun from scratch.
- ``upstream`` suites: the study's ``run.sh`` runs the benchmark's own code (its own
  process, at the suite's pinned commit) and writes each cell's outputs plus a
  ``harness.json`` sidecar. Each cell names that directory in ``results``. The verb
  checks the pinned commit, the settings the harness sent and (when ``seats`` is
  given) the models in each seat, then reads the outputs with the suite's reader.
  Changed outputs reopen a completed run (``should_resume``).

Settings resolve as model-card defaults (``model_generation``) < the benchmark's own
(the suite's ``generation``) < ``generation_overrides``; the suite fixes thinking.
Every cell shares the resolved settings, which the manifest records with the served
names, vLLM's adapter roots and sha256, harness versions and ingested files.

Only vLLM endpoints (and Inspect's offline mock model, for tests) are reachable, so
nothing here can spend money; GPU time is governed by the study's pod budget.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from marli.config import doc_field, runtime_field, to_dict
from marli.errors import BackendError, ConfigError
from marli.eval.external.readers import ReaderResult
from marli.eval.external.report import markdown, summarize
from marli.eval.external.serving import probe, request_fields, resolve_endpoint, resolve_settings
from marli.eval.external.spec import SUITES, ExternalSuite
from marli.handles import Handle, atomic_write_text, register_handle, sha256_file
from marli.registry import FnRegistry
from marli.rundir import RunDir

_READERS: FnRegistry = FnRegistry("external eval readers")
_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


@dataclass
class ExternalCell:
    label: str = ""
    policy: str | None = doc_field(None, help="inspect suites: vllm ref of the evaluated model")
    seats: list[str] = doc_field(
        default_factory=list,
        help="upstream suites: served model per seat; checked against the harness outputs",
    )
    results: str | None = doc_field(
        None, help="upstream suites: the cell's harness output directory (from the study run.sh)"
    )
    focal: str | None = doc_field(
        None,
        help="multi-seat suites: report seat rows as focal (this model's seats) vs others",
    )


@dataclass
class ExternalEvalConfig:
    suite: str = doc_field("", help="external eval registry entry (marli list external_evals)")
    cells: list[ExternalCell] = doc_field(
        default_factory=list,
        help="{label, policy} (inspect) or {label, results, seats, focal} (upstream) per cell",
    )
    baseline: str | None = doc_field(None, help="cell label gains are measured against")
    model_generation: dict[str, Any] = doc_field(
        default_factory=dict,
        help="model-card sampling and chat_template_kwargs; the suite's own settings win",
    )
    generation_overrides: dict[str, Any] = doc_field(
        default_factory=dict, help="explicit deviations from the suite's settings (recorded)"
    )
    task_args: dict[str, Any] = doc_field(
        default_factory=dict, help="inspect suites: overrides of the suite's task_args"
    )
    epochs: int | None = None
    limit: int | None = doc_field(None, help="inspect suites: first N samples only (smoke runs)")
    max_connections: int = runtime_field(32, help="inspect: concurrent requests per cell")
    max_error_rate: float = runtime_field(
        0.05, help="fail when a cell's harness/model error rate exceeds this"
    )
    timeout_s: float = runtime_field(30.0, help="server probe timeout")
    request_timeout_s: int = runtime_field(
        1800, help="inspect: per-request timeout (long thinking at high concurrency)"
    )

    def __post_init__(self) -> None:
        labels = [cell.label for cell in self.cells]
        if len(set(labels)) != len(labels):
            raise ConfigError("cell labels must be unique")
        for label in labels:
            if not label or label in {".", ".."} or set(label) - _SAFE:
                raise ConfigError(f"cell label {label!r}: use letters, digits, '.', '_' or '-'")
        if self.baseline is not None and self.baseline not in labels:
            raise ConfigError(f"baseline {self.baseline!r} is not a cell label")
        if self.epochs is not None and (type(self.epochs) is not int or self.epochs < 1):
            raise ConfigError("epochs must be a positive integer")
        if self.limit is not None and (type(self.limit) is not int or self.limit < 1):
            raise ConfigError("limit must be a positive integer")
        if type(self.max_connections) is not int or self.max_connections < 1:
            raise ConfigError("max_connections must be a positive integer")
        if not 0 <= self.max_error_rate <= 1:
            raise ConfigError("max_error_rate must be in [0, 1]")


@register_handle
@dataclass(frozen=True)
class ExternalEval(Handle):
    KIND: ClassVar[str] = "external_eval"
    MANIFEST: ClassVar[str] = "external.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("rows", "results", "markdown")

    suite: str
    rows: str
    results: str
    markdown: str
    n_cells: int

    def summary(self) -> dict[str, Any]:
        return {"suite": self.suite, "n_cells": self.n_cells}


def _output_digests(path: str) -> dict[str, str]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    return {str(p.resolve()): sha256_file(p) for p in files}


def should_resume(handle: ExternalEval, cfg: ExternalEvalConfig) -> bool:
    """Reread upstream outputs when any of them changed since the report."""
    recorded = handle.meta.get("cells", {})
    for cell in cfg.cells:
        if cell.results is not None:
            current = _output_digests(cell.results) if Path(cell.results).exists() else {}
            if current != recorded.get(cell.label, {}).get("outputs"):
                return True
    return False


def _check_cells(suite: ExternalSuite, cfg: ExternalEvalConfig) -> None:
    if not cfg.cells:
        raise ConfigError("eval external needs at least one cell")
    for cell in cfg.cells:
        if suite.kind == "inspect":
            if cell.policy is None or cell.results is not None or cell.seats:
                raise ConfigError(f"cell {cell.label!r}: inspect suites take policy only")
            resolve_endpoint(cell.policy)
        elif cell.results is None or cell.policy is not None:
            raise ConfigError(f"cell {cell.label!r}: upstream suites take results (and seats)")
    if suite.kind == "inspect" and (cfg.task_args.keys() - suite.task_args.keys()):
        unknown = sorted(cfg.task_args.keys() - suite.task_args.keys())
        raise ConfigError(
            f"unknown task_args {unknown}; the suite declares {sorted(suite.task_args)}"
        )


def _check_harness(
    suite: ExternalSuite, cell: ExternalCell, result: ReaderResult, request: dict[str, Any]
) -> None:
    commit = result.harness.get("commit")
    if commit != suite.source["commit"]:
        raise ConfigError(
            f"cell {cell.label!r}: harness ran commit {commit!r}, suite pins "
            f"{suite.source['commit']!r}"
        )
    sent = result.harness.get("request")
    if sent != request:
        raise ConfigError(
            f"cell {cell.label!r}: the harness sent {sent!r}, the resolved settings are {request!r}"
        )
    if cell.seats:
        for assignment in result.harness.get("seat_assignments", []):
            if Counter(assignment) != Counter(cell.seats):
                raise ConfigError(
                    f"cell {cell.label!r}: a run seated {assignment}, expected {cell.seats}"
                )


def focal_rows(rows: list[dict[str, Any]], focal: str) -> list[dict[str, Any]]:
    """Regroup per-seat rows (condition ``[<prefix>/]seat<k>``) as focal vs others.

    In a mixed cell, ``focal`` rows are the seats running ``focal`` and ``others`` the
    rest. In a homogeneous cell (every seat runs one model, e.g. the all-base baseline)
    each seat row counts for both, so mixed cells have a like-for-like comparison.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    passthrough = []
    for item in rows:
        prefix, _, seat = item["condition"].rpartition("/")
        if seat.startswith("seat") and seat[4:].isdigit():
            grouped.setdefault((item["sample"], prefix), []).append(item)
        else:
            passthrough.append(item)
    out = list(passthrough)
    for (_, prefix), seats in grouped.items():
        models = {item["detail"].get("model") for item in seats}
        lead = f"{prefix}/" if prefix else ""
        for item in seats:
            mine = item["detail"].get("model") == focal
            groups = ["focal", "others"] if len(models) == 1 else ["focal" if mine else "others"]
            out.extend({**item, "condition": lead + group} for group in groups)
    return out


def _notes(label: str, rows: list[dict[str, Any]], max_error_rate: float) -> list[str]:
    notes = []
    n = len(rows)
    errors = sum(item["error"] is not None for item in rows)
    if n and errors / n > max_error_rate:
        raise BackendError(f"cell {label!r}: {errors}/{n} rows failed (max_error_rate)")
    if errors:
        notes.append(f"{label}: {errors} of {n} rows are harness/model errors (excluded)")
    leaked = sum(bool(item["detail"].get("reasoning_in_content")) for item in rows)
    if leaked:
        notes.append(
            f"{label}: {leaked} replies carried reasoning markup in the answer text; serve "
            "with vLLM's reasoning parser"
        )
    missing = sum(value is None for item in rows for value in item["metrics"].values())
    values = sum(len(item["metrics"]) for item in rows)
    if values and missing / values > 0.1:
        notes.append(f"{label}: {missing}/{values} decisions unparsed (>10%)")
    return notes


async def external(cfg: ExternalEvalConfig, run: RunDir) -> ExternalEval:
    suite = SUITES.load(cfg.suite)
    _check_cells(suite, cfg)
    settings = resolve_settings(suite, cfg.model_generation, cfg.generation_overrides)
    request = request_fields(settings)
    reader = _READERS.get(suite.reader)
    task_args = {**suite.task_args, **cfg.task_args}
    epochs = cfg.epochs if cfg.epochs is not None else suite.epochs
    cells: dict[str, list[dict[str, Any]]] = {}
    records: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    for cell in cfg.cells:
        record_path = run.path(f"cells/{cell.label}.json")
        rows_path = run.path(f"cells/{cell.label}.rows.jsonl")
        if suite.kind == "inspect":
            if record_path.exists():  # finished on an earlier attempt
                records[cell.label] = json.loads(record_path.read_text(encoding="utf-8"))
                cells[cell.label] = [
                    json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
                ]
                notes.extend(_notes(cell.label, cells[cell.label], cfg.max_error_rate))
                continue
            from marli.eval.external.inspect_run import run_cell

            assert cell.policy is not None
            endpoint = resolve_endpoint(cell.policy)
            served = await probe(endpoint, timeout_s=cfg.timeout_s)
            log_dir = run.path(f"logs/{cell.label}")
            for stale in log_dir.glob("*.eval"):  # an interrupted attempt's partial log
                stale.unlink()
            await run_cell(
                suite,
                endpoint,
                settings,
                task_args=task_args,
                epochs=epochs,
                limit=cfg.limit,
                log_dir=log_dir,
                max_connections=cfg.max_connections,
                max_error_rate=cfg.max_error_rate,
                request_timeout_s=cfg.request_timeout_s,
            )
            result = reader(log_dir, suite)
            record = {"policy": cell.policy, "served": served, "harness": result.harness}
            record["logs"] = {
                str(Path(path).relative_to(run.out)): digest
                for path, digest in result.files.items()
            }
        else:
            assert cell.results is not None
            result = reader(Path(cell.results), suite)
            _check_harness(suite, cell, result, request)
            record = {
                "results": str(Path(cell.results).resolve()),
                "seats": list(cell.seats),
                "harness": result.harness,
                "outputs": _output_digests(cell.results),
            }
        rows = result.rows if cell.focal is None else focal_rows(result.rows, cell.focal)
        rows = [{**item, "cell": cell.label} for item in rows]
        notes.extend(_notes(cell.label, rows, cfg.max_error_rate))
        record["n_rows"] = len(rows)
        atomic_write_text(rows_path, "".join(json.dumps(item) + "\n" for item in rows))
        atomic_write_text(record_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
        cells[cell.label] = rows
        records[cell.label] = record
    results = summarize(cells, suite, cfg.baseline)
    atomic_write_text(
        run.path("rows.jsonl"),
        "".join(json.dumps(item) + "\n" for rows in cells.values() for item in rows),
    )
    atomic_write_text(
        run.path("results.jsonl"),
        "".join(json.dumps(_finite(item), sort_keys=True) + "\n" for item in results),
    )
    atomic_write_text(run.path("RESULTS.md"), markdown(results, suite, cfg.baseline, notes))
    for note in notes:
        warnings.warn(note, stacklevel=2)
    return ExternalEval(
        root=run.out,
        suite=suite.name,
        rows="rows.jsonl",
        results="results.jsonl",
        markdown="RESULTS.md",
        n_cells=len(cfg.cells),
        meta={
            "suite_spec": to_dict(suite),
            "settings": settings,
            "request": request,
            "task_args": task_args if suite.kind == "inspect" else None,
            "epochs": epochs if suite.kind == "inspect" else None,
            "baseline": cfg.baseline,
            "cells": records,
            "warnings": notes,
        },
    )


def _finite(item: dict[str, Any]) -> dict[str, Any]:
    """JSON has no NaN: replace non-finite floats with None."""
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in item.items()
    }
