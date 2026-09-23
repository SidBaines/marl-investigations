"""Shared failure categories keep library errors aligned with CLI exit codes."""

from __future__ import annotations


class MarliError(Exception):
    """An unexpected or otherwise unclassified marli failure."""

    exit_code: int = 1


class ConfigError(MarliError, ValueError):
    """Invalid configuration or usage."""

    exit_code: int = 2


class DirtyTreeError(MarliError):
    """Training provenance cannot be tied to a clean commit."""

    exit_code: int = 2


class HashMismatchError(MarliError):
    """An existing run directory belongs to a different configuration."""

    exit_code: int = 3


class BudgetExceededError(MarliError):
    """A configured spending limit was exceeded."""

    exit_code: int = 4


class BackendError(MarliError):
    """A backend failed or cannot support the requested operation."""

    exit_code: int = 5
