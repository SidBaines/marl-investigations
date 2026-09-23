from __future__ import annotations

from pathlib import Path

import pytest

from marli.errors import ConfigError
from marli.interact.configs import PROTOCOL_CONFIGS, build_protocol, resolve_protocol
from marli.interact.protocols.single import SingleConfig, SingleProtocol
from marli.registry import catalog_names


def test_single_entry_and_registered_name_resolve_to_the_same_protocol() -> None:
    assert "single" in PROTOCOL_CONFIGS.names()
    assert "single" in catalog_names("protocols")
    for ref in ("single",):
        name, config, plain = resolve_protocol(ref)
        assert name == "single" and config == SingleConfig()
        assert plain == {"system_prompt": SingleConfig().system_prompt, "env_tools": []}


def test_overrides_merge_and_tuples_survive() -> None:
    protocol = build_protocol("single", {"env_tools": ["python"]})
    assert isinstance(protocol, SingleProtocol)
    assert protocol.config.env_tools == ("python",)
    assert protocol.roles()[0].tools == ("submit", "python")


def test_unknown_keys_and_refs_are_config_errors() -> None:
    with pytest.raises(ConfigError, match="bogus"):
        resolve_protocol("single", {"bogus": 1})
    with pytest.raises(ConfigError, match="unknown protocol"):
        resolve_protocol("no_such_protocol")


def test_every_config_entry_builds() -> None:
    for name in PROTOCOL_CONFIGS.names():
        build_protocol(name)


def test_entry_files_are_named_by_stem() -> None:
    directory = Path(PROTOCOL_CONFIGS.directory)
    assert all(path.stem in PROTOCOL_CONFIGS.names() for path in directory.glob("*.yaml"))


def test_from_mappings_retuples_nested_dataclasses_in_lists() -> None:
    from dataclasses import dataclass, field

    from marli.config import from_mappings

    @dataclass
    class Inner:
        names: tuple[str, ...] = ()

    @dataclass
    class Outer:
        items: list[Inner] = field(default_factory=list)
        tags: tuple[str, ...] = ()

    out = from_mappings(Outer, {"items": [{"names": ["a"]}], "tags": ["x"]})
    assert out.items[0].names == ("a",) and out.tags == ("x",)
