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
        raise ConfigError("local training backend not implemented until M4")
    raise ConfigError(f"unknown training backend {name!r}; expected tinker, fake or local")
