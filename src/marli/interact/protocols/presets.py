"""Fixed swarm controls make the independent-sampling baseline reproducible."""

from __future__ import annotations

from dataclasses import dataclass

from marli.errors import ConfigError
from marli.interact.protocols.swarm import SwarmConfig, SwarmProtocol
from marli.interact.system import PROTOCOLS
from marli.interact.workspace import DeliverySpec


@dataclass(frozen=True)
class IndependentConfig(SwarmConfig):
    delivery: DeliverySpec = DeliverySpec(mode="pull")
    peer_tools: tuple[str, ...] = ("submit",)
    peer_system_prompt: str = (
        "You are {agent_id}. Solve the task independently. Use submit to give your final answer."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.delivery.mode != "pull"
            or tuple(self.peer_tools) != ("submit",)
            or {"read_scratchpad", "list_scratchpads"}.intersection(self.env_tools)
            or self.aggregation != "vote"
            or self.stop_on_consensus
            or self.auto_publish is not None
        ):
            raise ConfigError("independent requires pull delivery, submit only, and a full vote")


@PROTOCOLS.register("independent")
class IndependentProtocol(SwarmProtocol):
    name = "independent"
    config_type = IndependentConfig

    def __init__(self, config: IndependentConfig | None = None) -> None:
        super().__init__(config if config is not None else IndependentConfig())
