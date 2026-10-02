"""Keep contributor contexts and credit separate while the env workspace persists.

Roles are declared before the environment is available to the protocol, so the
tool list is config-driven: slotted envs (e.g. code_rules) supply graded
submissions through their slot hooks and need no ``submit``; non-slotted envs
must list the built-in ``submit`` in ``tools`` (checked when the episode runs).

``opener_role`` gives slot 0 its own role (same tools and prompt), so it can be
seated on a different policy, e.g. a frozen checkpoint, while later slots train.
Agent ids stay ``contrib<slot>`` either way.
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
    opener_role: str | None = None

    def __post_init__(self) -> None:
        if type(self.n_agents) is not int or self.n_agents < 1:
            raise ConfigError("n_agents must be a positive integer")
        if self.opener_role is not None:
            if not isinstance(self.opener_role, str) or not self.opener_role.isidentifier():
                raise ConfigError("opener_role must be an identifier-like role name")
            if self.opener_role == "contributor":
                raise ConfigError("opener_role must differ from 'contributor'")
            if self.n_agents < 2:
                raise ConfigError("opener_role needs n_agents >= 2")


@PROTOCOLS.register("relay")
class RelayProtocol(Protocol):
    name = "relay"
    config_type = RelayConfig

    def __init__(self, config: RelayConfig | None = None) -> None:
        self.config = config if config is not None else RelayConfig()

    def roles(self) -> list[RoleSpec]:
        tools = tuple(dict.fromkeys((*self.config.tools, *self.config.env_tools)))
        opener = self.config.opener_role
        roles = [
            RoleSpec(
                "contributor",
                tools,
                self.config.system_prompt,
                count=self.config.n_agents - (opener is not None),
                graded=True,
            )
        ]
        if opener is not None:
            prompt = self.config.system_prompt
            roles.insert(0, RoleSpec(opener, tools, prompt, count=1, graded=True))
        return roles

    def _role(self, slot: int) -> str:
        return self.config.opener_role if slot == 0 and self.config.opener_role else "contributor"

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
        if not slotted and "submit" not in self.config.tools:
            raise ConfigError("relay with a non-slotted env needs 'submit' in tools")
        submissions: dict[str, str | None] = {}
        final_answer: str | None = None
        for slot in range(self.config.n_agents):
            agent_id = f"contrib{slot}"
            role = self._role(slot)
            if slotted:
                io.env.bind_agent(agent_id, slot)
                first_message = io.env.slot_message(slot)
            else:
                first_message = io.env.task_message(role)
            handle = await io.start_agent(
                role,
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
