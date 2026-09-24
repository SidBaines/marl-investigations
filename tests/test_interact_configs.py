from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from _marli_code_fixtures import code_task

from marli.config import from_mappings
from marli.envs.code_fn import CodeFnEnv
from marli.errors import ConfigError
from marli.interact.configs import PROTOCOL_CONFIGS, build_protocol, resolve_protocol
from marli.interact.protocols.single import SingleConfig, SingleProtocol
from marli.interact.system import SystemIO
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


@dataclass
class _Inner:
    names: tuple[str, ...] = ()


@dataclass
class _Outer:
    items: list[_Inner] = field(default_factory=list)
    tags: tuple[str, ...] = ()


def test_from_mappings_retuples_nested_dataclasses_in_lists() -> None:
    out = from_mappings(_Outer, {"items": [{"names": ["a"]}], "tags": ["x"]})
    assert out.items[0].names == ("a",) and out.tags == ("x",)


def test_nested_frozen_defaults_accept_overrides() -> None:
    _, config, plain = resolve_protocol("swarm_n4_push", {"delivery": {"view": "full"}})
    assert config.delivery.mode == "push" and config.delivery.view == "full"
    assert plain["delivery"]["view"] == "full"


@pytest.mark.parametrize("name", ["swarm", "independent_n4", "debate_n3_r2"])
@pytest.mark.parametrize("n_agents", [1, 2])
async def test_code_vote_presets_fail_before_starting_agents(name: str, n_agents: int) -> None:
    protocol = build_protocol(name, {"n_agents": n_agents})
    start = AsyncMock()
    io = cast(SystemIO, SimpleNamespace(env=CodeFnEnv({}, code_task()), start_agent=start))
    with pytest.raises(ConfigError, match="vote.*code_fn"):
        await protocol.run(io)
    start.assert_not_awaited()
