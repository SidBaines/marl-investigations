"""Fixed swarm controls define independent sampling and repeated debate baselines."""

from __future__ import annotations

from dataclasses import dataclass, replace

from marli.errors import ConfigError
from marli.interact.limits import Limits
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


@dataclass(frozen=True)
class DebateConfig(SwarmConfig):
    rounds: int = 2
    delivery: DeliverySpec = DeliverySpec(mode="push")
    peer_tools: tuple[str, ...] = ()
    auto_publish: str | None = "final_text"
    aggregation: str = "vote"
    peer_system_prompt: str = (
        "You are {agent_id}, one of {n_agents} peers ({peer_ids}). "
        "Debate the task over {rounds} rounds. In each round, explain your reasoning "
        "and give a final answer in \\boxed{{}} in your visible reply. "
        "From round 2 you will see the other peers' previous replies; check their "
        "reasoning and revise your answer if needed. Your last round's reply is "
        "your final submission."
    )

    def __post_init__(self) -> None:
        if type(self.rounds) is not int or self.rounds < 1:
            raise ConfigError("rounds must be a positive integer")
        super().__post_init__()
        if (
            self.delivery.mode != "push"
            or self.peer_tools
            or self.env_tools
            or self.auto_publish != "final_text"
            or self.aggregation != "vote"
            or self.stop_on_consensus
        ):
            raise ConfigError("debate requires push delivery, no tools, final_text and a full vote")

    def prompt_fields(self) -> dict[str, str | int]:
        return {**super().prompt_fields(), "rounds": self.rounds}


@PROTOCOLS.register("debate")
class DebateProtocol(SwarmProtocol):
    name = "debate"
    config_type = DebateConfig

    def __init__(self, config: DebateConfig | None = None) -> None:
        super().__init__(config if config is not None else DebateConfig())

    def adjust_limits(self, limits: Limits) -> Limits:
        limits = super().adjust_limits(limits)
        return replace(
            limits,
            on_no_tool_call="final_text_continue",
            episode=replace(limits.episode, max_ticks=self.config.rounds),
        )
