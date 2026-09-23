"""Each grid cell owns resumable child runs; the grid manifest commits the final report.

The runner only hashes top-level input fields. Nested TaskSet paths are hashed
as configuration here, and their manifest digests are hashed by child rollouts.
Use a new grid output directory when replacing a TaskSet manifest in place.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from marli.config import from_mappings, runtime_field, to_dict
from marli.errors import ConfigError
from marli.eval.policies import PolicySpec
from marli.eval.report import Report, ReportConfig, build_report
from marli.eval.rollout import RolloutConfig
from marli.eval.score import ScoreConfig, Scores
from marli.rundir import RunDir
from marli.verbs import run_verb


@dataclass
class RolloutDefaults(RolloutConfig):
    # Nested input metadata is forbidden by run_verb; child rollouts resolve this input.
    tasks: str | None = None


@dataclass
class CellSpec:
    label: str
    tasks: str | None = None
    protocol: str | None = None
    protocol_config: dict[str, Any] = field(default_factory=dict)
    policies: dict[str, PolicySpec] = field(default_factory=dict)
    seating: dict[str, str] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    episodes_per_task: int | None = None

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


@dataclass
class GridConfig:
    cells: list[CellSpec] = field(default_factory=list)
    common: RolloutDefaults = field(default_factory=RolloutDefaults)
    baseline: str | None = None
    parallel_cells: int = runtime_field(1, help="concurrent cells; each has its own spend guard")

    def __post_init__(self) -> None:
        labels = [cell.label for cell in self.cells]
        if len(set(labels)) != len(labels):
            raise ConfigError("grid cell labels must be unique")
        if self.baseline is not None and self.baseline not in labels:
            raise ConfigError(f"unknown baseline cell {self.baseline!r}")
        if type(self.parallel_cells) is not int or self.parallel_cells < 1:
            raise ConfigError("parallel_cells must be a positive integer")


async def grid(cfg: GridConfig, run: RunDir) -> Report:
    if not cfg.cells:
        raise ConfigError("grid requires at least one cell")
    configs: dict[str, RolloutConfig] = {}
    for cell in cfg.cells:
        overrides = {
            key: value
            for key, value in to_dict(cell).items()
            if key != "label" and value is not None
        }
        configs[cell.label] = from_mappings(RolloutConfig, to_dict(cfg.common), overrides)
        if configs[cell.label].tasks is None:
            raise ConfigError(f"cell {cell.label!r} requires tasks (in cell or common)")
    semaphore = asyncio.Semaphore(cfg.parallel_cells)
    sources: dict[str, Scores] = {}
    stopped: Exception | None = None

    async def run_cell(cell: CellSpec) -> None:
        nonlocal stopped
        async with semaphore:
            if stopped is not None:
                return
            try:
                out = run.path(f"cells/{cell.label}")
                episodes = await run_verb("eval rollout", configs[cell.label], out=out / "rollout")
                scores = await run_verb(
                    "eval score", ScoreConfig(str(episodes.manifest)), out=out / "score"
                )
                sources[cell.label] = scores.handle
            except Exception as exc:
                stopped = exc

    workers = [asyncio.create_task(run_cell(cell)) for cell in cfg.cells]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    if stopped is not None:
        raise stopped
    return build_report(
        {cell.label: sources[cell.label] for cell in cfg.cells},
        ReportConfig(baseline=cfg.baseline),
        run,
    )
