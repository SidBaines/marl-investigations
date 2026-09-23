"""Named protocol configurations: one YAML per entry in this directory.

Each file is ``{protocol: <registered protocol name>, config: {...}, notes: "..."}``
and its name is the filename stem (``marli list protocols``). A protocol class
declares its config dataclass as the class attribute ``config_type`` (e.g.
``SingleProtocol.config_type = SingleConfig``). :func:`resolve_protocol` merges
defaults < the entry's ``config`` < caller ``overrides`` into that dataclass
(unknown keys raise ``ConfigError``) and :func:`build_protocol` instantiates it.

A reference is either a config entry name (``swarm_n4``) or a registered
protocol name (``single``) with its default config; entry names win.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marli.config import from_mappings, to_dict
from marli.errors import ConfigError
from marli.registry import Registry


@dataclass(frozen=True)
class ProtocolEntry:
    name: str
    protocol: str
    config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


PROTOCOL_CONFIGS: Registry[ProtocolEntry] = Registry(
    "protocols", Path(__file__).parent, ProtocolEntry
)


def load_protocol_config(name: str) -> ProtocolEntry:
    return PROTOCOL_CONFIGS.load(name)


def resolve_protocol(
    ref: str, overrides: Mapping[str, Any] | None = None
) -> tuple[str, Any, dict[str, Any]]:
    """Return ``(protocol name, config instance, plain resolved config dict)``."""
    from marli.interact.system import PROTOCOLS, load_builtin_protocols

    load_builtin_protocols()
    if ref in PROTOCOL_CONFIGS.names():
        entry = load_protocol_config(ref)
        name, base = entry.protocol, entry.config
    elif ref in PROTOCOLS.names():
        name, base = ref, {}
    else:
        raise ConfigError(
            f"unknown protocol {ref!r}; config entries: {PROTOCOL_CONFIGS.names()}, "
            f"registered protocols: {PROTOCOLS.names()}"
        )
    factory = PROTOCOLS.get(name)
    config_type = getattr(factory, "config_type", None)
    if config_type is None:
        if base or overrides:
            raise ConfigError(f"protocol {name!r} declares no config_type; it takes no config")
        return name, None, {}
    config = from_mappings(config_type, base, overrides or {})
    return name, config, to_dict(config)


def build_protocol(ref: str, overrides: Mapping[str, Any] | None = None) -> Any:
    """Resolve ``ref`` (entry or protocol name) + overrides into a Protocol instance."""
    from marli.interact.system import PROTOCOLS

    name, config, _ = resolve_protocol(ref, overrides)
    return PROTOCOLS.get(name)(config)
