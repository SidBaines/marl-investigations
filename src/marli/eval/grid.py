"""Cells own scientific identity and spend; the grid assembles their report.

Policy overrides merge field by field over common, including sampling fields,
just like limits. An omitted override inherits its common value.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from marli.budget import SpendGuard
from marli.config import (
    compose,
    config_hash,
    doc_field,
    from_mappings,
    input_field,
    runtime_field,
    to_dict,
)
from marli.errors import BudgetExceededError, ConfigError
from marli.eval.policies import PolicySpec
from marli.eval.report import Report, ReportConfig, build_report
from marli.eval.rollout import EpisodeSet, RolloutConfig
from marli.eval.score import ScoreConfig, Scores
from marli.handles import sha256_file
from marli.rundir import RunDir
from marli.verbs import input_manifest, resolve_inputs, run_verb


@dataclass
class RolloutDefaults(RolloutConfig):
    tasks: str | None = input_field(None, help="common TaskSet manifest or dir")
    regrade: bool = False


@dataclass
class CellSpec:
    label: str
    tasks: str | None = input_field(None, help="cell TaskSet manifest or dir")
    scores: str | None = input_field(None, help="existing Scores instead of sampling")
    protocol: str | None = None
    protocol_config: dict[str, Any] = field(default_factory=dict)
    # Sparse mappings retain omitted fields, unlike default-filled PolicySpecs.
    policies: dict[str, Any] = doc_field(
        default_factory=dict, help="sparse overrides merge field by field over common policies"
    )
    seating: dict[str, str] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    episodes_per_task: int | None = None
    regrade: bool | None = None

    def __post_init__(self) -> None:
        if (
            not self.label
            or self.label in {".", ".."}
            or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for char in self.label
            )
        ):
            raise ConfigError("cell labels must be safe directory names: letters, digits, . _ -")
        if self.scores is not None and (
            self.tasks is not None
            or self.protocol is not None
            or self.protocol_config
            or self.policies
            or self.seating
            or self.limits
            or self.episodes_per_task is not None
            or self.regrade is not None
        ):
            raise ConfigError("scores cells cannot also specify rollout or regrade settings")


@dataclass
class GridConfig:
    cells: list[CellSpec] = runtime_field(
        default_factory=list,
        help="independently hashed cells; sparse policy and limit fields merge over common",
    )
    common: RolloutDefaults = runtime_field(default_factory=RolloutDefaults)
    baseline: str | None = None
    cells_dir: str | None = runtime_field(None, help="cell storage; defaults to <out>/cells")
    force_cells: list[str] = runtime_field(default_factory=list, help="labels to rerun with force")
    max_usd: float | None = runtime_field(
        None, help="total grid spend, including previous attempts"
    )
    parallel_cells: int = runtime_field(1, help="concurrent cells sharing remaining grid spend")

    def __post_init__(self) -> None:
        labels = [cell.label for cell in self.cells]
        if len(set(labels)) != len(labels):
            raise ConfigError("grid cell labels must be unique")
        if self.baseline is not None and self.baseline not in labels:
            raise ConfigError(f"unknown baseline cell {self.baseline!r}")
        if set(self.force_cells) - set(labels):
            raise ConfigError("force_cells must name grid cell labels")
        if type(self.parallel_cells) is not int or self.parallel_cells < 1:
            raise ConfigError("parallel_cells must be a positive integer")
        SpendGuard(self.max_usd)


def rollout_config(common: RolloutDefaults, cell: CellSpec) -> RolloutConfig:
    """Merge sparse cell policy/limit overrides before validating the resolved rollout."""
    defaults = to_dict(common)
    defaults.pop("regrade")
    overrides = {
        key: value
        for key, value in to_dict(cell).items()
        if key not in {"label", "scores", "regrade"} and value is not None
    }
    # Python callers can still pass complete PolicySpec records; YAML uses sparse dicts.
    overrides["policies"] = {
        name: to_dict(spec) if isinstance(spec, PolicySpec) else spec
        for name, spec in cell.policies.items()
    }
    return from_mappings(RolloutConfig, defaults, overrides)


def recorded_spend(out: Path) -> float:
    """Prefer finalized cost; incomplete runs retain all charged attempts in progress."""
    if (out / "episodes.json").exists():
        return EpisodeSet.load(out).cost_usd
    if (out / "progress.json").exists():
        return float(json.loads((out / "progress.json").read_text())["spend"]["spent_usd"])
    return 0.0


def _record_version(path: Path) -> list[int] | None:
    """Identify a local run record across force's atomic replacement, even at the same hash."""
    if not path.exists():
        return None
    stat = path.stat()
    return [stat.st_ino, stat.st_mtime_ns]


def cell_state(cfg: GridConfig, out: Path) -> dict[str, Any]:
    """Snapshot labels, child identities and manifests for report invalidation.

    Labels are refreshable report membership, allowing additions without changing
    the parent's immutable run hash. Scientific settings belong to each cell.
    """
    root = Path(cfg.cells_dir).resolve() if cfg.cells_dir else out / "cells"
    state = {}
    for cell in cfg.cells:
        if cell.scores is not None:
            path = input_manifest(cell.scores, "scores")
            state[cell.label] = {"scores": str(path), "digest": sha256_file(path)}
            continue
        config, digests = resolve_inputs(rollout_config(cfg.common, cell))
        directory = root / cell.label
        state[cell.label] = {
            "directory": str(directory),
            "rollout_hash": config_hash(config, input_digests=digests),
            "regrade": cfg.common.regrade if cell.regrade is None else cell.regrade,
            "manifests": {
                name: sha256_file(directory / name) if (directory / name).exists() else None
                for name in ("rollout/episodes.json", "score/scores.json")
            },
        }
    return state


def should_resume(handle: Report, cfg: GridConfig) -> bool:
    """Reopen only when cell membership, science, results or retry state changed."""
    state = cell_state(cfg, handle.root)
    if cfg.force_cells or state != handle.meta.get("cells"):
        return True
    cost = sum(handle.meta.get("retired_cost_usd", {}).values())
    for cell in cfg.cells:
        if cell.scores is not None:
            cost += float(Scores.load(state[cell.label]["scores"]).meta.get("cost_usd", 0))
            continue
        path = Path(state[cell.label]["directory"]) / "rollout"
        if not (path / "episodes.json").exists():
            return True
        episodes = EpisodeSet.load(path)
        cost += episodes.cost_usd
        if cfg.common.retry_failed and episodes.n_failed:
            return True
    return cfg.max_usd is not None and cost > cfg.max_usd


def force_changed_cells(cfg: GridConfig, out: Path) -> GridConfig:
    """Translate grid --force into child replacements, preserving unchanged cells."""
    forced = set(cfg.force_cells)
    for label, state in cell_state(cfg, out).items():
        if "directory" not in state:
            continue
        root = Path(state["directory"])
        checks = [(root / "rollout", state["rollout_hash"])]
        if (root / "rollout" / "episodes.json").exists():
            score_cfg, digests = resolve_inputs(
                ScoreConfig(str(root / "rollout" / "episodes.json"), regrade=state["regrade"])
            )
            checks.append((root / "score", config_hash(score_cfg, input_digests=digests)))
        for directory, expected in checks:
            record = directory / ".marli" / "run.json"
            if record.exists() and json.loads(record.read_text())["config_hash"] != expected:
                forced.add(label)
    return replace(cfg, force_cells=sorted(forced))


async def grid(cfg: GridConfig, run: RunDir) -> Report:
    if not cfg.cells:
        raise ConfigError("grid requires at least one cell")
    cells_dir = Path(cfg.cells_dir).resolve() if cfg.cells_dir else run.path("cells")
    configs: dict[str, RolloutConfig] = {}
    sources: dict[str, Scores] = {}
    costs: dict[str, float] = {}
    child_warnings: list[str] = []
    retired: dict[str, float] = {}
    forced_attempts: dict[str, dict[str, Any]] = {}
    if run.path("progress.json").exists():
        saved_progress = json.loads(run.path("progress.json").read_text())
        retired = saved_progress.get("retired_cost_usd", {})
        # A force can be interrupted after removing the old child run but before
        # recording its replacement. Keep those already spent dollars reserved.
        for label, attempt in saved_progress.get("forced_attempts", {}).items():
            if _record_version(Path(attempt["record"])) != attempt["version"]:
                retired[label] = retired.get(label, 0.0) + attempt["cost_usd"]
    for cell in cfg.cells:
        if cell.scores is not None:
            sources[cell.label] = Scores.load(input_manifest(cell.scores, "scores"))
            costs[cell.label] = float(sources[cell.label].meta.get("cost_usd", 0.0))
        else:
            configs[cell.label] = rollout_config(cfg.common, cell)
            if configs[cell.label].tasks is None:
                raise ConfigError(f"cell {cell.label!r} requires tasks (in cell or common)")
            costs[cell.label] = recorded_spend(cells_dir / cell.label / "rollout")

    def remaining() -> float | None:
        if cfg.max_usd is None:
            return None
        available = cfg.max_usd - sum(costs.values()) - sum(retired.values())
        if available < -1e-12:
            raise BudgetExceededError(f"grid budget ${cfg.max_usd:.4f} exceeded")
        return max(0.0, available)

    def progress() -> None:
        run.write_progress(
            {
                "cost_usd": sum(costs.values()) + sum(retired.values()),
                "cells": costs,
                "retired_cost_usd": retired,
                "forced_attempts": forced_attempts,
            }
        )

    remaining()

    async def run_cell(cell: CellSpec, allocation: float | None) -> None:
        out = cells_dir / cell.label
        forced = cell.label in cfg.force_cells
        prior = costs[cell.label]
        record = out / "rollout" / ".marli" / "run.json"
        original_record = _record_version(record)
        config = configs[cell.label]
        # Rollout's guard is cumulative, so a resumed cell keeps its historical
        # spend in its ceiling; the newly allocated share is only incremental.
        ceiling = None if allocation is None else (0.0 if forced else prior) + allocation
        if config.max_usd is not None:
            ceiling = config.max_usd if ceiling is None else min(config.max_usd, ceiling)
        if forced and prior:
            forced_attempts[cell.label] = {
                "record": str(record),
                "version": original_record,
                "cost_usd": prior,
            }
            progress()
        try:
            episodes = await run_verb(
                "eval rollout",
                replace(config, max_usd=ceiling),
                out=out / "rollout",
                force=forced,
            )
            child_warnings.extend(episodes.warnings)
            score_cfg = ScoreConfig(
                str(episodes.manifest),
                regrade=cfg.common.regrade if cell.regrade is None else cell.regrade,
            )
            score_out = out / "score"
            refresh = False
            if (score_out / ".marli" / "run.json").exists():
                previous = compose(ScoreConfig, score_out / "config.yaml")
                resolved, digests = resolve_inputs(score_cfg)
                recorded = json.loads((score_out / ".marli" / "run.json").read_text())
                # Retried episodes invalidate even an interrupted score run. A
                # changed regrade setting still requires explicit cell force.
                refresh = previous.regrade == score_cfg.regrade and recorded[
                    "config_hash"
                ] != config_hash(resolved, input_digests=digests)
            scores = await run_verb("eval score", score_cfg, out=score_out, force=forced or refresh)
            child_warnings.extend(scores.warnings)
            sources[cell.label] = scores.handle
        finally:
            if forced and _record_version(record) != original_record:
                retired[cell.label] = retired.get(cell.label, 0.0) + prior
            forced_attempts.pop(cell.label, None)
            costs[cell.label] = recorded_spend(out / "rollout")
            progress()

    pending = []
    for cell in cfg.cells:
        if cell.scores is not None:
            continue
        manifest = cells_dir / cell.label / "rollout" / "episodes.json"
        if (
            cell.label in cfg.force_cells
            or not manifest.exists()
            or (configs[cell.label].retry_failed and EpisodeSet.load(manifest).n_failed)
        ):
            pending.append(cell)
        else:
            # Completed cells still validate identity and refresh scores, but do
            # not reserve a share of dollars needed by the remaining samplers.
            await run_cell(cell, 0.0)
    # Fixed waves reserve equal shares before launching. Unused shares are released
    # at the next wave; no concurrent cell can allocate the same dollars twice.
    for start in range(0, len(pending), cfg.parallel_cells):
        wave = pending[start : start + cfg.parallel_cells]
        available = remaining()
        share = None if available is None else available / len(wave)

        outcomes = await asyncio.gather(
            *(run_cell(cell, share) for cell in wave), return_exceptions=True
        )
        errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if errors:
            raise next(
                (error for error in errors if isinstance(error, BudgetExceededError)), errors[0]
            )
        remaining()
    report = build_report(
        {cell.label: sources[cell.label] for cell in cfg.cells},
        ReportConfig(baseline=cfg.baseline),
        run,
    )
    return replace(
        report,
        meta={
            **report.meta,
            "cells": cell_state(cfg, run.out),
            "retired_cost_usd": retired,
            "warnings": list(dict.fromkeys([*child_warnings, *report.meta["warnings"]])),
        },
    )
