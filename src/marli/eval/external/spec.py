"""External eval suites: one YAML per benchmark, naming its harness, pin and reading.

A suite records what the benchmark itself specifies. ``generation`` holds only the
settings the benchmark defines (anything it leaves unset falls back to the policy's
model-card settings supplied by the run config). ``thinking`` is fixed per suite and
must be identical across every cell of a run.

Rows (what readers return, one per sample x condition)::

    {"sample": str,          # pairing unit across cells (a task id; a trial id if pair=none)
     "condition": str,       # e.g. game name or HiddenBench profile
     "metrics": {name: float | None},   # None = the decision could not be parsed
     "parse_failures": int,  # decisions in this row that could not be parsed
     "error": str | None,    # harness/model error for this sample (not a parse failure)
     "detail": {...}}        # reader-specific diagnostics, kept out of the statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marli.registry import Registry

KINDS = ("inspect", "upstream")
METRIC_KINDS = ("binary", "share")  # binary: 0/1 decisions; share: a fraction in [0, 1]
PAIRINGS = ("sample", "none")  # sample: same tasks across cells; none: i.i.d. trials
THINKING = ("on", "off")
# Request settings a suite or run may set; anything else is a typo (unknown keys raise).
GENERATION_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
    "chat_template_kwargs",
)


def check_generation(settings: dict[str, Any], where: str) -> None:
    if not isinstance(settings, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = settings.keys() - set(GENERATION_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown generation keys {sorted(unknown)}")
    kwargs = settings.get("chat_template_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        raise ValueError(f"{where}: chat_template_kwargs must be a mapping")


@dataclass(frozen=True)
class ExternalSuite:
    name: str
    kind: str  # inspect | upstream
    description: str
    citation: str
    license: str
    # Pinned source: inspect: {package, version}; upstream: {repo, commit}.
    source: dict[str, str]
    reader: str  # module:function -> ReaderResult (see readers.py)
    metrics: list[dict[str, str]]  # [{name, kind, pair, label}]
    task: str | None = None  # inspect: module:function returning an inspect Task
    task_args: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)  # the benchmark's own settings
    thinking: str = "on"
    epochs: int = 1
    # Whether benchmark task text may be committed (contamination/licence); readers and
    # reports never copy prompts or transcripts into rows either way.
    commit_text: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.kind == "inspect":
            if not self.task or ":" not in self.task:
                raise ValueError("inspect suites need task: module:function")
            if self.source.keys() != {"package", "version"}:
                raise ValueError("inspect suites pin source: {package, version}")
        else:
            if self.task is not None or self.task_args:
                raise ValueError(
                    "upstream suites run outside marli; task/task_args are inspect-only"
                )
            if self.source.keys() != {"repo", "commit"}:
                raise ValueError("upstream suites pin source: {repo, commit}")
            commit = self.source["commit"]
            if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
                raise ValueError("upstream source.commit must be a full 40-character git sha")
        if ":" not in self.reader:
            raise ValueError("reader must be module:function")
        check_generation(self.generation, "generation")
        if "chat_template_kwargs" in self.generation:
            raise ValueError("thinking is set by `thinking`, not generation.chat_template_kwargs")
        if self.thinking not in THINKING:
            raise ValueError(f"thinking must be one of {THINKING}")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not self.metrics:
            raise ValueError("a suite reports at least one metric")
        names = set()
        for metric in self.metrics:
            if not isinstance(metric, dict) or metric.keys() - {"name", "kind", "pair", "label"}:
                raise ValueError("metrics entries take name, kind, pair and label")
            if metric.get("kind") not in METRIC_KINDS or metric.get("pair") not in PAIRINGS:
                raise ValueError(f"metric kind in {METRIC_KINDS}, pair in {PAIRINGS}")
            if not metric.get("name") or metric["name"] in names:
                raise ValueError("metric names must be unique and non-empty")
            names.add(metric["name"])

    def metric(self, name: str) -> dict[str, str]:
        return next(metric for metric in self.metrics if metric["name"] == name)


SUITES: Registry[ExternalSuite] = Registry(
    "external_evals", Path(__file__).parent / "suites", ExternalSuite
)
