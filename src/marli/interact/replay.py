"""Replay a saved episode's prefix exactly, then sample live past its cut.

Continuing an episode must be distributed exactly as if it had been run with the
larger budget from the start. Prompts never state the budget, so everything an
agent saw before the cut is unchanged: replayed calls return their recorded
completions, and replayed tool calls run for real (to rebuild the workspace and
env state) while the agent receives the *recorded* result text. At the cut the
model keeps going; after it, every call of every agent is live.

The cut is global and requires sequential agents (``single``, ``relay``,
``multi_session``): the first agent whose turn ended on a limit
(:data:`TRUNCATING_ENDS`). Earlier agents replay completely. That agent replays
up to its last non-forced call; later agents run live, since they saw a repo
without the truncated agent's remaining work. At that last call:

* a completion without a stop token (cut at its allocation) is *extended* when
  the new limits allow more tokens (sample with prompt = buffer + recorded ids,
  return recorded + new ids);
* otherwise it replays, and the next call is live.

Trailing forced calls (``on_exhaust=force_final``) are dropped: they were the
harness's reaction to the exhaustion that the new limits remove.

Equivalence checks, per replayed call (a failure is a *divergence*):

* the prompt length equals the recorded ``prompt_len``, and the purpose matches;
* the allocation is consistent with the recording: a stop-terminated completion
  needs ``max_tokens >= len``, and a length-cut one needs ``max_tokens == len``
  (otherwise the new limits changed where an earlier completion ended);
* the parsed tool calls match the recorded ones by position and name, and each
  runs without a backend error.

Live calls use fresh seeds, ``derive_seed(plan seed, "continue", agent seed)``,
so that continued randomness is independent of the source's.
``on_divergence="fail"`` ends the episode not-ok with a clear error;
``"live"`` switches every later call to live and flags it.

Replayed and recorded tool results are compared after normalising volatile
fields (durations, sandbox paths, truncation counts). A mismatch is only
counted: the agent always sees the recorded text.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from marli.errors import BackendError, ConfigError
from marli.interact.types import Call, Episode, EventKind, Purpose
from marli.seeds import derive_seed

TRUNCATING_ENDS = frozenset({"budget", "ctx", "max_ticks"})
SEQUENTIAL_PROTOCOLS = frozenset({"single", "relay", "multi_session"})
_FORCED = (Purpose.FINAL, Purpose.REPORT)


class ReplayDivergence(BackendError):
    """The continuation no longer reproduces the recorded prefix."""


@dataclass(frozen=True)
class Cut:
    agent_id: str
    last_call: int  # index of the truncated agent's last replayable (non-forced) call; -1: none
    reason: str  # e.g. "budget (agent.max_gen_tokens)"


@dataclass(frozen=True)
class Directive:
    kind: str  # "replay" | "cut" | "live"
    call: Call | None = None


def call_index(call: Call) -> int:
    return int(call.call_id.rsplit("/c", 1)[1])


def agent_ends(episode: Episode) -> dict[str, str | None]:
    return {
        event.agent_id: event.data.get("ended_by")
        for event in episode.events
        if event.kind == EventKind.DONE
    }


def agent_order(episode: Episode) -> list[str]:
    """Agents in the order they started (first recorded event)."""
    first: dict[str, int] = {}
    for event in episode.events:
        first.setdefault(event.agent_id, event.seq)
    known = [agent.agent_id for agent in episode.agents]
    return sorted(known, key=lambda agent: first.get(agent, 1 << 62))


def find_cut(episode: Episode) -> Cut | None:
    """The first agent whose turn ended on a limit, or None."""
    ends = agent_ends(episode)
    for agent in agent_order(episode):
        ended_by = ends.get(agent)
        if ended_by not in TRUNCATING_ENDS:
            continue
        calls = sorted((call for call in episode.calls if call.agent_id == agent), key=call_index)
        while calls and calls[-1].forced and calls[-1].purpose in _FORCED:
            calls.pop()
        hits = episode.limits_hit.get(agent, ()) + episode.limits_hit.get("_episode", ())
        reason = f"{ended_by} ({', '.join(hits)})" if hits else str(ended_by)
        return Cut(agent, call_index(calls[-1]) if calls else -1, reason)
    return None


_DURATION = re.compile(r'"duration_s":\s*-?[0-9.eE+-]+')
_SANDBOX = re.compile(r"(?:/[\w.\-]+)*/marli-[a-z0-9_]{8}\b")
_TRUNCATED = re.compile(r"…\[\d+ chars truncated\]…")


def normalize_result(text: str) -> str:
    text = _DURATION.sub('"duration_s": 0', text)
    text = _SANDBOX.sub("<sandbox>", text)
    return _TRUNCATED.sub("…[N chars truncated]…", text)


class ReplayPlan:
    """Episode-level replay state shared by every agent of one continued episode."""

    def __init__(
        self,
        source: Episode,
        cut: Cut | None,
        *,
        seed: int,
        on_divergence: str = "fail",
    ) -> None:
        if on_divergence not in {"fail", "live"}:
            raise ConfigError("on_divergence must be fail or live")
        self.source = source
        self.cut = cut
        self.seed = seed
        self.on_divergence = on_divergence
        self.calls: dict[str, list[Call]] = defaultdict(list)
        for call in sorted(source.calls, key=call_index):
            self.calls[call.agent_id].append(call)
        order = agent_order(source)
        if cut is None:
            self.full = set(order)
        else:
            self.full = set(order[: order.index(cut.agent_id)])
        self.consumed: dict[str, int] = defaultdict(int)
        self.divergence: str | None = None
        self.errors: list[str] = []
        self.replayed_calls = 0
        self.extended_tokens = 0
        self.extended = False
        self.tool_calls_replayed = 0
        self.tool_mismatches = 0

    @property
    def live(self) -> bool:
        return self.divergence is not None

    def directive(self, agent_id: str, index: int) -> Directive:
        if self.divergence is not None:
            if self.on_divergence == "fail":
                raise ReplayDivergence(f"after an earlier divergence: {self.divergence}")
            return Directive("live")
        recorded = self.calls.get(agent_id, [])
        if agent_id in self.full:
            if index >= len(recorded):
                self.diverge(f"{agent_id} made more calls than recorded ({len(recorded)})")
                return Directive("live")
            return Directive("replay", recorded[index])
        if self.cut is not None and agent_id == self.cut.agent_id and index <= self.cut.last_call:
            kind = "cut" if index == self.cut.last_call else "replay"
            return Directive(kind, recorded[index])
        return Directive("live")

    def consume(self, agent_id: str) -> None:
        self.consumed[agent_id] += 1
        self.replayed_calls += 1

    def live_seed(self, seed: int) -> int:
        return derive_seed(self.seed, "continue", seed)

    def check(
        self,
        recorded: Call,
        *,
        prompt_len: int,
        purpose: Purpose,
        max_tokens: int,
        stopped: bool,
        cut: bool,
    ) -> str | None:
        """Why this recorded call cannot be replayed here, or None."""
        n = len(recorded.completion_ids)
        if purpose != recorded.purpose:
            return f"{recorded.call_id}: purpose {purpose.value} != recorded {recorded.purpose}"
        if prompt_len != recorded.prompt_len:
            return (
                f"{recorded.call_id}: prompt length {prompt_len} != recorded {recorded.prompt_len}"
            )
        if max_tokens < n:
            return f"{recorded.call_id}: allocation {max_tokens} < recorded completion {n}"
        if not stopped and max_tokens != n and not cut:
            return (
                f"{recorded.call_id}: the new limits allow {max_tokens} tokens where the "
                f"recorded completion was cut at {n}; the prefix would not be equivalent"
            )
        return None

    def diverge(self, message: str) -> None:
        """Record a divergence; in fail mode, end the calling agent with a backend error.

        The agent's error carries the message into the episode's errors.
        """
        if self.divergence is None:
            self.divergence = message
        if self.on_divergence == "fail":
            raise ReplayDivergence(message)

    def finish(self) -> list[str]:
        """Check that every replayed agent consumed its recorded prefix; return errors."""
        if self.divergence is None:
            for agent_id, recorded in self.calls.items():
                expected = (
                    len(recorded)
                    if agent_id in self.full
                    else (
                        self.cut.last_call + 1 if self.cut and agent_id == self.cut.agent_id else 0
                    )
                )
                if self.consumed[agent_id] < expected:
                    message = (
                        f"{agent_id} ended after {self.consumed[agent_id]} of {expected} "
                        "recorded calls"
                    )
                    self.divergence = message
                    if self.on_divergence == "fail":
                        self.errors.append(f"replay diverged: {message}")
                    break
        return list(dict.fromkeys(self.errors))

    def summary(self) -> dict[str, Any]:
        return {
            "cut": None
            if self.cut is None
            else {
                "agent_id": self.cut.agent_id,
                "call_index": self.cut.last_call,
                "reason": self.cut.reason,
                "extended": self.extended,
            },
            "replayed_calls": self.replayed_calls,
            "extended_tokens": self.extended_tokens,
            "tool_calls_replayed": self.tool_calls_replayed,
            "tool_mismatches": self.tool_mismatches,
            "diverged": self.divergence,
        }
