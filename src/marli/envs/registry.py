"""Environment factories keep episode construction independent of concrete envs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from marli.registry import FnRegistry

if TYPE_CHECKING:
    from marli.envs.base import Env, Task

ENVS = FnRegistry("envs")


def load_builtin_envs() -> None:
    """Register built-ins once via Python's import cache, before any named lookup."""
    from marli.envs import code_fn as _code_fn  # noqa: F401
    from marli.envs import code_rules as _code_rules  # noqa: F401
    from marli.envs import math as _math  # noqa: F401


def make_env(name: str, config: dict[str, Any], task: Task) -> Env:
    load_builtin_envs()
    return ENVS.get(name)(config, task)
