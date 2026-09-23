"""Typed configuration from scimt, with reproducible hashes and CLI schemas.

Composition applies defaults < YAML files from left to right < dotted overrides.
Prefer plain str fields with __post_init__ validation over typing.Literal or
frozenset, which OmegaConf 2.3 rejects. Config Enum members must have name ==
value so accepted strings match serialized values. __post_init__ methods must
be idempotent validation: never derive or mutate fields, since normalization
for hashing re-runs them. Runtime knobs and input manifest paths are excluded
from hashes; callers supply input content digests separately.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import types
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml
from omegaconf import MISSING, OmegaConf
from omegaconf.errors import MissingMandatoryValue, OmegaConfBaseException

from marli.errors import ConfigError


def compose[T](
    cls: type[T], *yaml_paths: str | Path, overrides: list[str] | tuple[str, ...] = ()
) -> T:
    """Merge configuration layers and instantiate the validated dataclass."""
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise TypeError(f"compose() takes a dataclass type, got {cls!r}")
    _validate_enums(cls)
    layers = []
    for path in yaml_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        try:
            layers.append(OmegaConf.load(path))
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        except (OmegaConfBaseException, ValueError) as exc:
            raise _config_error(exc) from exc
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(cls), *layers, OmegaConf.from_dotlist(list(overrides))
        )
        obj = OmegaConf.to_object(merged)
    except (OmegaConfBaseException, ValueError) as exc:
        raise _config_error(exc) from exc
    assert isinstance(obj, cls)
    return obj


def _config_error(exc: OmegaConfBaseException | ValueError) -> ConfigError:
    if isinstance(exc, OmegaConfBaseException) and exc.full_key:
        if isinstance(exc, MissingMandatoryValue):
            return ConfigError(f"missing mandatory config key '{exc.full_key}'")
        return ConfigError(f"invalid config key '{exc.full_key}': {str(exc).splitlines()[0]}")
    return ConfigError(str(exc))


def _validate_enums(annotation: Any, name: str = "") -> None:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if any(member.name != member.value for member in annotation):
            raise TypeError(
                f"config field '{name}': enum {annotation.__name__} members must have name == value"
            )
    elif isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        hints = get_type_hints(annotation)
        for field in dataclasses.fields(annotation):
            field_name = f"{name}.{field.name}" if name else field.name
            _validate_enums(hints[field.name], field_name)
    else:
        for arg in get_args(annotation):
            _validate_enums(arg, name)


def parse[T](cls: type[T], argv: list[str] | tuple[str, ...] | None = None) -> T:
    """Split scimt-style positional YAML paths and dotted key=value overrides."""
    args = sys.argv[1:] if argv is None else argv
    paths = [arg for arg in args if "=" not in arg]
    overrides = [arg for arg in args if "=" in arg]
    return compose(cls, *paths, overrides=overrides)


def from_mappings[T](cls: type[T], *layers: Any) -> T:
    """Like :func:`compose`, from in-memory mappings (defaults < layers, left to right).

    Used for configs nested in registry entries (e.g. a protocol YAML's ``config``
    block). Unknown keys raise ``ConfigError``; ``tuple``-annotated fields come
    back as tuples (OmegaConf returns lists).
    """
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise TypeError(f"from_mappings() takes a dataclass type, got {cls!r}")
    _validate_enums(cls)
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(cls), *(OmegaConf.create(dict(layer or {})) for layer in layers)
        )
        obj = OmegaConf.to_object(merged)
    except (OmegaConfBaseException, ValueError) as exc:
        raise _config_error(exc) from exc
    assert isinstance(obj, cls)
    return _retuple(obj)


def _retuple(obj: Any) -> Any:
    if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
        return obj
    hints = get_type_hints(type(obj))
    changes: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, list) and _is_tuple(hints.get(f.name)):
            changes[f.name] = tuple(value)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            fixed = _retuple(value)
            if fixed is not value:
                changes[f.name] = fixed
    return dataclasses.replace(obj, **changes) if changes else obj


def _is_tuple(annotation: Any) -> bool:
    if annotation is tuple or get_origin(annotation) is tuple:
        return True
    return any(_is_tuple(arg) for arg in get_args(annotation) if arg is not type(None))


def save(cfg: Any, path: str | Path) -> Path:
    """Save a dataclass instance as a YAML configuration."""
    _require_instance(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(cfg), path)
    return path


def runtime_field(
    default: Any = dataclasses.MISSING,
    *,
    default_factory: Any = dataclasses.MISSING,
    help: str | None = None,
) -> Any:
    """Declare a runtime knob that does not affect configuration identity."""
    return dataclasses.field(
        default=default,
        default_factory=default_factory,
        metadata={"runtime": True, "help": help},
    )


def input_field(default: Any = dataclasses.MISSING, *, help: str | None = None) -> Any:
    """Declare an input manifest path whose content digest is hashed separately."""
    return dataclasses.field(default=default, metadata={"input": True, "help": help})


def doc_field(
    default: Any = dataclasses.MISSING,
    *,
    default_factory: Any = dataclasses.MISSING,
    help: str,
) -> Any:
    """Attach CLI documentation to an ordinary configuration field."""
    return dataclasses.field(
        default=default, default_factory=default_factory, metadata={"help": help}
    )


def _require_instance(cfg: Any) -> None:
    if isinstance(cfg, type) or not dataclasses.is_dataclass(cfg):
        raise TypeError(f"expected a dataclass instance, got {cfg!r}")


def _plain(value: Any, *, hashable: bool = False) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name), hashable=hashable)
            for field in dataclasses.fields(value)
            if not (hashable and (field.metadata.get("runtime") or field.metadata.get("input")))
        }
    if isinstance(value, Enum):
        return _plain(value.value, hashable=hashable)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item, hashable=hashable) for item in value]
    if isinstance(value, dict):
        return {
            _plain(key, hashable=hashable): _plain(item, hashable=hashable)
            for key, item in value.items()
        }
    return value


def to_dict(cfg: Any) -> dict[str, Any]:
    """Convert a dataclass instance to plain JSON-compatible containers."""
    _require_instance(cfg)
    return _plain(cfg)


def hashable_dict(cfg: Any) -> dict[str, Any]:
    """Convert configuration while recursively omitting runtime and input fields."""
    _require_instance(cfg)
    return _plain(cfg, hashable=True)


def config_hash(cfg: Any, *, input_digests: Iterable[str] = ()) -> str:
    """Hash normalized scientific settings and the sorted input content digests.

    Normalization re-runs __post_init__, which must be idempotent validation
    and must never derive or mutate fields.
    """
    _require_instance(cfg)
    try:
        normalized = OmegaConf.to_object(OmegaConf.structured(cfg))
    except (OmegaConfBaseException, ValueError) as exc:
        raise _config_error(exc) from exc
    payload = json.dumps(
        {"config": hashable_dict(normalized), "inputs": sorted(input_digests)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _type_name(annotation: Any) -> str:
    if annotation is Ellipsis:
        return "..."
    if annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, Union):
        return " | ".join(_type_name(arg) for arg in args)
    if origin is not None:
        return f"{_type_name(origin)}[{', '.join(_type_name(arg) for arg in args)}]"
    return getattr(annotation, "__name__", str(annotation))


def describe_config(cls: type[Any]) -> list[dict[str, Any]]:
    """Describe leaf fields in declaration order, including effective defaults."""
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise TypeError(f"describe_config() takes a dataclass type, got {cls!r}")
    _validate_enums(cls)
    return _describe(cls)


def _describe(
    cls: type[Any],
    *,
    prefix: str = "",
    defaults: Any = dataclasses.MISSING,
    inherited_runtime: bool = False,
    inherited_input: bool = False,
    inherited_required: bool = False,
) -> list[dict[str, Any]]:
    hints = get_type_hints(cls)
    result = []
    for field in dataclasses.fields(cls):
        annotation = hints[field.name]
        name = f"{prefix}{field.name}"
        is_runtime = inherited_runtime or bool(field.metadata.get("runtime"))
        is_input = inherited_input or bool(field.metadata.get("input"))
        if inherited_required:
            default = dataclasses.MISSING
        elif dataclasses.is_dataclass(defaults) and not isinstance(defaults, type):
            default = getattr(defaults, field.name)
        elif field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        else:
            default = field.default
        required = default is dataclasses.MISSING or (
            isinstance(default, str) and default == MISSING
        )
        if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
            result.extend(
                _describe(
                    annotation,
                    prefix=f"{name}.",
                    defaults=default,
                    inherited_runtime=is_runtime,
                    inherited_input=is_input,
                    inherited_required=required,
                )
            )
        else:
            result.append(
                {
                    "name": name,
                    "type": _type_name(annotation),
                    "default": None if required else _plain(default),
                    "required": required,
                    "runtime": is_runtime,
                    "input": is_input,
                    "help": field.metadata.get("help"),
                }
            )
    return result
