"""Small reproducible registries, following scimt's filename/name invariant.

YAML entries hold scalars, lists, and plain dictionaries; nested-dataclass
conversion is deliberately out of scope. Function references and the catalog
resolve lazily so enumeration does not pull in unused backend dependencies.
Field types are not validated beyond __post_init__; entries should validate
critical fields there.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from marli.errors import ConfigError


class Registry[T]:
    """One YAML file per entry: <directory>/<name>.yaml -> cls(**data)."""

    def __init__(
        self, kind: str, directory: str | Path, cls: type[T], *, name_field: str = "name"
    ) -> None:
        if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
            raise TypeError(f"Registry requires a dataclass type, got {cls!r}")
        self.kind = kind
        self.directory = Path(directory)
        self.cls = cls
        self.name_field = name_field

    def names(self) -> list[str]:
        """Return available YAML entry names in sorted order."""
        return sorted(path.stem for path in self.directory.glob("*.yaml") if path.is_file())

    def path(self, name: str) -> Path:
        """Resolve an entry by name or report the available choices."""
        names = self.names()
        if name not in names:
            raise ConfigError(
                f"unknown {self.kind} registry entry {name!r}; available names: {names}"
            )
        return self.directory / f"{name}.yaml"

    def load(self, name: str) -> T:
        """Load one entry, rejecting ambiguous names and invalid dataclass keys."""
        path = self.path(name)
        context = f"{self.kind} registry entry {name!r}"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{context}: invalid YAML: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ConfigError(f"{context}: expected a YAML mapping")
        fields = {field.name: field for field in dataclasses.fields(self.cls) if field.init}
        unknown = sorted(data.keys() - fields.keys(), key=str)
        if unknown:
            raise ConfigError(f"{context}: unknown keys {unknown}")
        if self.name_field in fields:
            if self.name_field in data and data[self.name_field] != path.stem:
                raise ConfigError(
                    f"{context}: {self.name_field} {data[self.name_field]!r} "
                    f"must match filename stem {path.stem!r}"
                )
            data[self.name_field] = path.stem
        required = [
            field.name
            for field in fields.values()
            if field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
            and field.name not in data
        ]
        if required:
            raise ConfigError(f"{context}: missing required fields {required}")
        try:
            return self.cls(**data)
        except ValueError as exc:
            raise ConfigError(f"{context}: {exc}") from exc

    def load_all(self) -> dict[str, T]:
        """Load every entry in sorted name order."""
        return {name: self.load(name) for name in self.names()}


class FnRegistry[F: Callable[..., Any]]:
    """Named functions (rewards, advantage estimators, protocols). Never lambdas."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._functions: dict[str, F] = {}

    def register(self, name: str) -> Callable[[F], F]:
        """Register a named callable without changing the decorated function."""

        def decorator(function: F) -> F:
            if name in self._functions:
                raise ValueError(f"{self.kind} function {name!r} is already registered")
            if not callable(function) or getattr(function, "__name__", None) == "<lambda>":
                raise ValueError(f"{self.kind} function {name!r} must be a named callable")
            self._functions[name] = function
            return function

        return decorator

    def get(self, ref: str) -> F:
        """Resolve a registered name or import a module:attr callable lazily."""
        if ref in self._functions:
            return self._functions[ref]
        if ":" not in ref:
            raise ConfigError(
                f"unknown {self.kind} function {ref!r}; available names: {self.names()}"
            )
        try:
            function = _resolve(ref)
        except (ImportError, AttributeError, ValueError) as exc:
            raise ConfigError(f"cannot resolve {self.kind} function {ref!r}: {exc}") from exc
        if not callable(function) or getattr(function, "__name__", None) == "<lambda>":
            raise ConfigError(f"{self.kind} function {ref!r} must be a named callable")
        return function

    def names(self) -> list[str]:
        """Return sorted registered names."""
        return sorted(self._functions)


def _resolve(ref: str) -> Any:
    module, attribute = ref.split(":", 1)
    return getattr(importlib.import_module(module), attribute)


CATALOG: dict[str, str] = {
    "models": "marli.model:MODELS",
    "tasks": "marli.tasks.source:SOURCES",
}


def catalog_kinds() -> list[str]:
    """Return catalog kinds without importing their registries."""
    return sorted(CATALOG)


def catalog_names(kind: str) -> list[str]:
    """Resolve only the requested registry and enumerate its entries."""
    if kind not in CATALOG:
        raise ConfigError(f"unknown registry kind {kind!r}; available kinds: {catalog_kinds()}")
    return _resolve(CATALOG[kind]).names()
