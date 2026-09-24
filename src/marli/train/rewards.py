"""Auxiliary signals use recorded behavior, so scoring never reruns an agent."""

from __future__ import annotations

from collections.abc import Callable

from marli.interact.types import Episode, Purpose, ReadVia
from marli.registry import FnRegistry

AUX_REWARDS: FnRegistry[Callable[[Episode, str], float]] = FnRegistry("aux_rewards")


@AUX_REWARDS.register("format")
def format_reward(episode: Episode, agent_id: str) -> float:
    calls = [
        call for call in episode.calls if call.agent_id == agent_id and call.purpose == Purpose.ACT
    ]
    if not calls:
        return 1.0
    return sum(all(tool.parsed_ok for tool in call.tool_calls) for call in calls) / len(calls)


@AUX_REWARDS.register("submitted")
def submitted(episode: Episode, agent_id: str) -> float:
    return float(episode.outcome.submissions.get(agent_id) is not None)


@AUX_REWARDS.register("cross_reads")
def cross_reads(episode: Episode, agent_id: str) -> float:
    return float(
        sum(
            read.writer != agent_id and read.via == ReadVia.PULL
            for call in episode.calls
            if call.agent_id == agent_id
            for read in call.reads
        )
    )


@AUX_REWARDS.register("cross_reads_all")
def cross_reads_all(episode: Episode, agent_id: str) -> float:
    return float(
        sum(
            read.writer != agent_id
            for call in episode.calls
            if call.agent_id == agent_id
            for read in call.reads
        )
    )


@AUX_REWARDS.register("workers_spawned")
def workers_spawned(episode: Episode, agent_id: str) -> float:
    return float(sum(agent.parent == agent_id for agent in episode.agents))
