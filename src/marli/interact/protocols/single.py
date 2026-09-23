"""A single solver provides the baseline for every multi-agent comparison."""

from __future__ import annotations

from dataclasses import dataclass

from marli.interact.system import PROTOCOLS, Protocol, RoleSpec, SystemIO
from marli.interact.types import Outcome


@dataclass(frozen=True)
class SingleConfig:
    system_prompt: str = "Solve the task carefully. Use submit to provide your final answer."
    env_tools: tuple[str, ...] = ()


@PROTOCOLS.register("single")
class SingleProtocol(Protocol):
    name = "single"

    def __init__(self, config: SingleConfig | None = None) -> None:
        self.config = config if config is not None else SingleConfig()

    def roles(self) -> list[RoleSpec]:
        return [RoleSpec("solver", ("submit", *self.config.env_tools), self.config.system_prompt)]

    async def run(self, io: SystemIO) -> Outcome:
        handle = await io.start_agent(
            "solver",
            agent_id="solver0",
            seat_key=("solver", 0),
            first_message=io.env.task_message("solver"),
        )
        result = await handle
        return Outcome(result.submission, {"solver0": result.submission}, "single")
