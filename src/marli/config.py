"""Typed configuration from scimt, with reproducible hashes and CLI schemas.

Composition applies defaults < YAML files from left to right < dotted overrides.
Use str/Enum fields and __post_init__ validation instead of typing.Literal or
frozenset, which OmegaConf 2.3 rejects. Runtime knobs and input manifest paths
are excluded from hashes; callers supply input content digests separately.
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

from omegaconf import MISSING, OmegaConf
from omegaconf.errors import ConfigKeyError, OmegaConfBaseException

from marli.errors import ConfigError


def compose[T](
    cls: type[T], *yaml_paths: str | Path, overrides: list[str] | tuple[str, ...] = ()
) -> T:
    """Merge configuration layers and instantiate the validated dataclass."""
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise TypeError(f"compose() takes a dataclass type, got {cls!r}")
    try:
        layers = []
        for path in yaml_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"config file not found: {path}")
            layers.append(OmegaConf.load(path))
        merged = OmegaConf.merge(
            OmegaConf.structured(cls), *layers, OmegaConf.from_dotlist(list(overrides))
        )
        obj = OmegaConf.to_object(merged)
    except ConfigKeyError as exc:
        raise ConfigError(f"unknown config key '{exc.full_key}'") from exc
    except (OmegaConfBaseException, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    assert isinstance(obj, cls)
    return obj


def parse[T](cls: type[T], argv: list[str] | tuple[str, ...] | None = None) -> T:
    """Split scimt-style positional YAML paths and dotted key=value overrides."""
    args = sys.argv[1:] if argv is None else argv
    paths = [arg for arg in args if "=" not in arg]
    overrides = [arg for arg in args if "=" in arg]
    return compose(cls, *paths, overrides=overrides)


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
    """Hash canonical scientific settings and the sorted input content digests."""
    payload = json.dumps(
        {"config": hashable_dict(cfg), "inputs": sorted(input_digests)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _type_name(annotation: Any) -> str:
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
    return _describe(cls)


def _describe(
    cls: type[Any],
    *,
    prefix: str = "",
    defaults: Any = dataclasses.MISSING,
    runtime: bool = False,
    input: bool = False,
) -> list[dict[str, Any]]:
    hints = get_type_hints(cls)
    result = []
    for field in dataclasses.fields(cls):
        annotation = hints[field.name]
        name = f"{prefix}{field.name}"
        is_runtime = runtime or bool(field.metadata.get("runtime"))
        is_input = input or bool(field.metadata.get("input"))
        if dataclasses.is_dataclass(defaults) and not isinstance(defaults, type):
            default = getattr(defaults, field.name)
        elif field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        else:
            default = field.default
        if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
            result.extend(
                _describe(
                    annotation,
                    prefix=f"{name}.",
                    defaults=default,
                    runtime=is_runtime,
                    input=is_input,
                )
            )
        else:
            required = default is dataclasses.MISSING or (
                isinstance(default, str) and default == MISSING
            )
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
