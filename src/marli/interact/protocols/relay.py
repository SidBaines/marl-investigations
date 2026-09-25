"""Keep contributor contexts and credit separate while the env workspace persists.

Roles are declared before the environment is available to the protocol, so the
built-in submit tool is advertised for compatibility with non-slotted envs.
Slotted envs exclusively supply graded submissions through their slot hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from marli.errors import ConfigError
from marli.interact.limits import Limits
from marli.interact.system import PROTOCOLS, Protocol, RoleSpec, SystemIO
from marli.interact.types import Outcome


@dataclass(frozen=True)
class RelayConfig:
    n_agents: int = 4
    tools: tuple[str, ...] = ("end_session",)
    env_tools: tuple[str, ...] = ()
    system_prompt: str = (
        "You are a software engineer contributing to a shared code repository. "
        "Follow the instructions in the first message."
    )

    def __post_init__(self) -> None:
        if type(self.n_agents) is not int or self.n_agents < 1:
            raise ConfigError("n_agents must be a positive integer")


@PROTOCOLS.register("relay")
class RelayProtocol(Protocol):
    name = "relay"
    config_type = RelayConfig

    def __init__(self, config: RelayConfig | None = None) -> None:
        self.config = config if config is not None else RelayConfig()

    def roles(self) -> list[RoleSpec]:
        return [
            RoleSpec(
                "contributor",
                tuple(dict.fromkeys((*self.config.tools, "submit", *self.config.env_tools))),
                self.config.system_prompt,
                count=self.config.n_agents,
                graded=True,
            )
        ]

    def adjust_limits(self, limits: Limits) -> Limits:
        """Give each contributor its full budget without forcing a submission.

        Slotted envs have no built-in submit action that can force a CI result.
        Release the unused final reserve, and reject undersized episode budgets
        instead of silently changing the requested compute.
        """
        if limits.episode.max_gen_tokens < self.config.n_agents * limits.agent.max_gen_tokens:
            raise ConfigError("episode.max_gen_tokens must be >= n_agents * agent.max_gen_tokens")
        return replace(
            limits,
            agent=replace(limits.agent, final_reserve=0),
            session=replace(limits.session, max_sessions=1),
            on_exhaust="none",
        )

    async def run(self, io: SystemIO) -> Outcome:
        slotted = io.env.n_slots > 0
        if slotted and io.env.n_slots != self.config.n_agents:
            raise ConfigError("env.n_slots must equal relay n_agents")
        submissions: dict[str, str | None] = {}
        final_answer: str | None = None
        for slot in range(self.config.n_agents):
            agent_id = f"contrib{slot}"
            if slotted:
                io.env.bind_agent(agent_id, slot)
                first_message = io.env.slot_message(slot)
            else:
                first_message = io.env.task_message("contributor")
            handle = await io.start_agent(
                "contributor",
                agent_id=agent_id,
                seat_key=("contributor", slot),
                first_message=first_message,
            )
            result = await handle
            submission = io.env.slot_submission(agent_id) if slotted else result.submission
            submissions[agent_id] = submission
            if submission is not None:
                final_answer = submission
        if slotted:
            final_answer = io.env.bundle(submissions)
        return Outcome(final_answer, submissions, "relay")
