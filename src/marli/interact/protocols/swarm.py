"""Peers share versioned scratchpads while retaining independent answers and credit.

The runtime owns forced finalization at the episode's max_ticks (or stops
without an answer for on_exhaust=none). Aggregation waits for those results,
including exhaustion results whose ended_by is not 'submit'. A separate
finalizer reads the committed latest pads only after every peer has finished.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from string import Formatter

from marli.errors import ConfigError
from marli.interact.system import PROTOCOLS, AgentHandle, AgentResult, Protocol, RoleSpec, SystemIO
from marli.interact.types import Outcome
from marli.interact.workspace import DeliverySpec, Permissions


@dataclass(frozen=True)
class SwarmConfig:
    n_agents: int = 4
    delivery: DeliverySpec = DeliverySpec()
    peer_tools: tuple[str, ...] = (
        "write_scratchpad",
        "read_scratchpad",
        "list_scratchpads",
        "submit",
    )
    env_tools: tuple[str, ...] = ()
    aggregation: str = "vote"
    finalizer: str = "separate"
    allow_resubmit: bool = False
    stop_on_consensus: int = 0
    peer_system_prompt: str = (
        "You are {agent_id}, one of {n_agents} peers ({peer_ids}). "
        "Solve the task and share useful work through your own scratchpad. "
        "Use submit to give your final answer."
    )
    finalizer_system_prompt: str = (
        "You are {agent_id}. Resolve the work of {n_agents} peers ({peer_ids}) "
        "and use submit to give the final answer."
    )
    auto_publish: str | None = None

    def __post_init__(self) -> None:
        if type(self.n_agents) is not int or self.n_agents < 1:
            raise ConfigError("n_agents must be a positive integer")
        if self.aggregation not in {"vote", "finalizer"}:
            raise ConfigError("aggregation must be vote or finalizer")
        if self.finalizer not in {"separate", "peer0"}:
            raise ConfigError("finalizer must be separate or peer0")
        if self.allow_resubmit:
            raise ConfigError("allow_resubmit=True is not supported in v1")
        if self.finalizer == "peer0":
            raise ConfigError("finalizer='peer0' is not supported in v1")
        if (
            type(self.stop_on_consensus) is not int
            or not 0 <= self.stop_on_consensus <= self.n_agents
        ):
            raise ConfigError("stop_on_consensus must be an integer between 0 and n_agents")
        if self.auto_publish not in {None, "final_text"}:
            raise ConfigError("auto_publish must be None or final_text")
        self.delivery.__post_init__()
        for name in ("peer_system_prompt", "finalizer_system_prompt"):
            try:
                getattr(self, name).format(
                    agent_id="peer0", role="peer", n_agents=self.n_agents, peer_ids="peer0"
                )
            except (KeyError, IndexError, ValueError) as exc:
                raise ConfigError(f"invalid {name} template: {exc}") from exc


@PROTOCOLS.register("swarm")
class SwarmProtocol(Protocol):
    name = "swarm"
    config_type = SwarmConfig

    def __init__(self, config: SwarmConfig | None = None) -> None:
        self.config = config if config is not None else SwarmConfig()
        self.delivery = self.config.delivery

    @property
    def peer_ids(self) -> tuple[str, ...]:
        return tuple(f"peer{i}" for i in range(self.config.n_agents))

    def _prompt(self, template: str) -> str:
        # Resolve swarm-wide fields without consuming the runtime's per-agent
        # fields or literal braces (e.g. a boxed-answer instruction).
        parts = []
        for literal, name, spec, conversion in Formatter().parse(template):
            parts.append(literal.replace("{", "{{").replace("}", "}}"))
            if name is not None:
                field = "{" + name + ("!" + conversion if conversion else "")
                field += (":" + spec if spec else "") + "}"
                if name in {"n_agents", "peer_ids"}:
                    field = field.format(
                        n_agents=self.config.n_agents, peer_ids=", ".join(self.peer_ids)
                    )
                    field = field.replace("{", "{{").replace("}", "}}")
                parts.append(field)
        return "".join(parts)

    def roles(self) -> list[RoleSpec]:
        roles = [
            RoleSpec(
                "peer",
                (*self.config.peer_tools, *self.config.env_tools),
                self._prompt(self.config.peer_system_prompt),
                count=self.config.n_agents,
                permissions=Permissions(read_others=True, write_scratchpad=True),
                publish_final_text=self.config.auto_publish == "final_text",
            )
        ]
        if self.config.aggregation == "finalizer":
            roles.append(
                RoleSpec(
                    "finalizer",
                    ("submit",),
                    self._prompt(self.config.finalizer_system_prompt),
                    permissions=Permissions(read_others=True, write_scratchpad=False),
                )
            )
        return roles

    def _first_message(self, io: SystemIO, agent_id: str) -> str:
        others = ", ".join(peer for peer in self.peer_ids if peer != agent_id) or "none"
        preamble = (
            f"You are {agent_id}. Other peers: {others}. "
            "Each peer owns a scratchpad in the shared workspace. "
        )
        if self.config.delivery.mode == "pull" and not {
            "read_scratchpad",
            "list_scratchpads",
        }.intersection(self.config.peer_tools):
            preamble += "Solve independently; other peers' work is unavailable. "
        if "submit" in self.config.peer_tools:
            preamble += "Call submit(answer=...) when finished."
        else:
            preamble += "Give your answer in your visible final reply."
        return io.env.task_message("peer") + "\n\n" + preamble

    async def _wait_peers(self, io: SystemIO, handles: list[AgentHandle]) -> list[AgentResult]:
        if not self.config.stop_on_consensus:
            return await io.wait(handles)
        pending = list(handles)
        results: dict[str, AgentResult] = {}
        clusters: list[list[str]] = []
        while pending:
            done, _ = await asyncio.wait(
                [handle.task for handle in pending], return_when=asyncio.FIRST_COMPLETED
            )
            # Seat order stabilizes clustering when several peers finish together.
            for handle in pending:
                if handle.task not in done:
                    continue
                result = handle.task.result()
                results[result.agent_id] = result
                if result.submission is None:
                    continue
                for cluster in clusters:
                    if await io.env.same_answer(result.submission, cluster[0]):
                        cluster.append(result.submission)
                        break
                else:
                    clusters.append([result.submission])
            pending = [handle for handle in pending if handle.task not in done]
            if any(len(cluster) >= self.config.stop_on_consensus for cluster in clusters):
                for handle in pending:
                    io.stop(handle, "consensus")
                for result in await io.wait(pending):
                    results[result.agent_id] = result
                break
        return [results[handle.agent_id] for handle in handles]

    async def run(self, io: SystemIO) -> Outcome:
        handles = [
            await io.start_agent(
                "peer",
                agent_id=agent_id,
                seat_key=("peer", index),
                first_message=self._first_message(io, agent_id),
            )
            for index, agent_id in enumerate(self.peer_ids)
        ]
        results = await self._wait_peers(io, handles)
        submissions = {result.agent_id: result.submission for result in results}
        if self.config.aggregation == "vote":
            answer, votes = await io.vote(submissions)
            return Outcome(answer, submissions, "vote", votes)
        blocks = [io.env.task_message("finalizer"), "Final peer state:"]
        for peer in self.peer_ids:
            # EpisodeSystem exposes its workspace; own reads include an empty v0 pad.
            content, version = io.workspace.read(peer, peer)
            blocks.append(
                f"[{peer} scratchpad v{version}]\n{content}\n"
                f"[{peer} submission]\n"
                + (submissions[peer] if submissions[peer] is not None else "(no submission)")
            )
        result = await io.finalize(
            "finalizer", agent_id="finalizer0", first_message="\n\n".join(blocks)
        )
        return Outcome(result.submission, submissions, "finalizer")
