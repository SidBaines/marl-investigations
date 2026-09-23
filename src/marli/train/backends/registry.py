"""Select training backends without importing their optional SDKs at CLI startup."""

from __future__ import annotations

from typing import Any

from marli.budget import SpendGuard
from marli.errors import ConfigError
from marli.train.backends.base import TrainBackend


def make_backend(
    name: str, *, spend: SpendGuard | None, base_url: str | None = None, **kw: Any
) -> TrainBackend:
    if name == "fake":
        from marli.train.backends.fake import FakeBackend

        return FakeBackend(**kw)
    if name == "tinker":
        from marli.train.backends.tinker import TinkerBackend

        return TinkerBackend(spend, base_url=base_url, **kw)
    if name == "local":
        from marli.train.backends.local.backend import LocalBackend

        if {"server_json", "adapters_dir"} - kw.keys():
            raise ConfigError("local training requires server_json and adapters_dir")
        return LocalBackend(spend=spend, **kw)
    raise ConfigError(f"unknown training backend {name!r}; expected tinker, fake or local")
