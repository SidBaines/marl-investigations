"""Validate scientific settings before the loop constructs paid learners."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any
from urllib.parse import urlsplit

from marli.budget import SpendGuard
from marli.config import input_field, runtime_field
from marli.errors import ConfigError
from marli.eval.policies import SamplingOverrides
from marli.interact.limits import Limits
from marli.train.types import CreditConfig, LearnerSpec


@dataclass
class TrainRLConfig:
    tasks: str | None = input_field(None, help="training TaskSet")
    protocol: str = "single"
    protocol_config: dict[str, Any] = field(default_factory=dict)
    env: str = "math"
    env_config: dict[str, Any] = field(default_factory=dict)
    learners: dict[str, LearnerSpec] = field(default_factory=dict)
    seating: dict[str, str] = field(default_factory=dict)
    frozen_sampling: dict[str, SamplingOverrides] = field(default_factory=dict)
    credit: CreditConfig = field(default_factory=CreditConfig)
    limits: Limits = field(default_factory=Limits)
    schedule: str = "lockstep"
    batch_tasks: int = 8
    group_size: int = 4
    steps: int = 10
    checkpoint_every: int = 5
    seed: int = 0
    allow_idle: bool = False
    max_failed_frac: float = 0.5
    run_name: str = ""
    concurrency: int = runtime_field(16, help="concurrent episodes")
    max_usd: float | None = runtime_field(None, help="spend guard (sampling + training)")
    base_url: str | None = runtime_field(None, help="Tinker base URL (explicit)")

    def __post_init__(self) -> None:
        for name in ("batch_tasks", "group_size", "steps", "checkpoint_every", "concurrency"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ConfigError("seed must be an integer")
        if self.schedule not in {"lockstep", "async"}:
            raise ConfigError("schedule must be lockstep or async")
        if not isfinite(self.max_failed_frac) or not 0 <= self.max_failed_frac <= 1:
            raise ConfigError("max_failed_frac must be finite and in [0, 1]")
        for name in (self.run_name, *self.learners):
            if name and (
                name in {".", ".."}
                or any(
                    c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
                    for c in name
                )
            ):
                raise ConfigError("run and learner names must contain only letters, digits, _.-")
        if "" in self.learners:
            raise ConfigError("learner names must be non-empty")
        if self.base_url:
            url = urlsplit(self.base_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ConfigError("base_url must be an HTTP URL without credentials or query")
        for spec in self.learners.values():
            spec.__post_init__()
            if spec.loss == "cross_entropy":
                raise ConfigError("train rl does not support cross_entropy; use train sft")
            if spec.backend in {"tinker", "local"} and self.max_usd is None:
                raise ConfigError("paid learners require max_usd")
        for sampling in self.frozen_sampling.values():
            sampling.__post_init__()
        self.credit.__post_init__()
        self.limits.__post_init__()
        SpendGuard(self.max_usd)
