"""Swarm goldens pin shared-state timing, independent credit and aggregation."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import yaml
from _marli_test_envs import ArithEnv

from marli.errors import ConfigError
from marli.interact.limits import Limits
from marli.interact.protocols.presets import IndependentConfig, IndependentProtocol
from marli.interact.protocols.swarm import SwarmConfig, SwarmProtocol
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.scheduler import FakeClock
from marli.interact.system import PROTOCOLS
from marli.interact.types import EventKind, Purpose, ReadVia
from marli.interact.workspace import DeliverySpec, Permissions
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, turns_by_agent
from marli.render.fake import SPECIAL_IDS as S
from marli.render.fake import FakeRenderer


def submit(answer: str) -> Turn:
    return Turn(tool_calls=(("submit", {"answer": answer}),))


def swarm_spec(
    turns: dict[str, list[Turn]],
    *,
    protocol: SwarmProtocol | None = None,
    env: ArithEnv | None = None,
    limits: Limits | None = None,
    schedule: str = "lockstep",
) -> EpisodeSpec:
    env = env if env is not None else ArithEnv()
    renderer = FakeRenderer()
    protocol = protocol if protocol is not None else SwarmProtocol(SwarmConfig(n_agents=3))
    return EpisodeSpec(
        protocol,
        env,
        env.task,
        {role.role: "script" for role in protocol.roles()},
        {"script": ScriptedPolicy("script", renderer, turns_by_agent(renderer, turns))},
        {"script": FakeRenderer},
        limits if limits is not None else Limits(),
        schedule=schedule,
        run_seed=31,
    )


def shared_turns() -> dict[str, list[Turn]]:
    return {
        f"peer{i}": [
            Turn(tool_calls=(("write_scratchpad", {"content": f"work {i}\ndetail {i}"}),)),
            Turn(tool_calls=(("read_scratchpad", {"agent_id": f"peer{(i + 1) % 3}"}),)),
            submit("05" if i == 0 else "5"),
        ]
        for i in range(3)
    }


async def test_lockstep_shared_scratchpads_notify_pull_and_submit() -> None:
    spec = swarm_spec(shared_turns())
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.replayable
    assert episode.outcome.submissions == {"peer0": "05", "peer1": "5", "peer2": "5"}
    assert episode.outcome.final_answer == "05"
    assert episode.outcome.votes == {"5": 3} and episode.outcome.aggregation == "vote"
    assert [(agent.agent_id, agent.seat_key) for agent in episode.agents] == [
        (f"peer{i}", ("peer", i)) for i in range(3)
    ]
    assert [(write.writer, write.version, write.tick) for write in episode.workspace_log] == [
        (f"peer{i}", 1, 0) for i in range(3)
    ]
    policy = spec.policies["script"]
    for i in range(3):
        peer = f"peer{i}"
        calls = [call for call in episode.calls if call.agent_id == peer]
        prompts = [ctx for ctx in policy.calls if ctx.meta.agent_id == peer]
        assert [call.tick for call in calls] == [0, 1, 2]
        assert calls[0].reads == ()
        assert spec.task.prompt in prompts[0].prompt_text
        assert f"You are {peer}, one of 3 peers (peer0, peer1, peer2)" in prompts[0].prompt_text
        assert {(read.writer, read.version, read.via) for read in calls[1].reads} == {
            (f"peer{j}", 1, ReadVia.NOTIFY) for j in range(3) if j != i
        }
        assert f"[workspace] {peer} wrote" not in prompts[1].prompt_text
        assert f"detail {(i + 1) % 3}" in prompts[2].prompt_text
        assert calls[2].reads[0].via == ReadVia.PULL
        assert calls[2].reads[0].writer == f"peer{(i + 1) % 3}"
        for call, ctx in zip(calls, prompts, strict=True):
            buf = buffers[call.segment_id]
            assert tuple(buf[: call.prompt_len]) == ctx.prompt_ids
            assert tuple(buf[call.prompt_len : call.prompt_len + len(call.completion_ids)]) == (
                call.completion_ids
            )
    assert episode.grades == {
        agent: {"correct": 1.0} for agent in ("peer0", "peer1", "peer2", "_system")
    }


class FractionEnv(ArithEnv):
    async def same_answer(self, a: str | None, b: str | None) -> bool:
        return a is not None and b is not None and Fraction(a) == Fraction(b)


async def test_vote_uses_verifier_equivalence_not_canonical_equality() -> None:
    spec = swarm_spec(
        {"peer0": [submit("1/2")], "peer1": [submit("0.5")], "peer2": [submit("2")]},
        env=FractionEnv(),
    )
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "1/2"
    assert episode.outcome.votes == {"1/2": 2, "2": 1}
    assert episode.outcome.submissions == {"peer0": "1/2", "peer1": "0.5", "peer2": "2"}


@pytest.mark.parametrize("all_none", [False, True])
async def test_missing_submissions_never_win(all_none: bool) -> None:
    turns = {"peer0": [Turn("no answer")], "peer1": [Turn("no answer")]}
    turns["peer2"] = [Turn("no answer") if all_none else submit("5")]
    episode, _ = await run_episode(swarm_spec(turns, limits=Limits(on_no_tool_call="end_agent")))
    assert episode.outcome.final_answer == (None if all_none else "5")
    assert episode.outcome.votes == ({} if all_none else {"5": 1})
    assert episode.outcome.submissions["peer0"] is None
    assert episode.grades["peer0"] == {"correct": 0.0}


async def test_separate_finalizer_gets_task_latest_pads_and_submissions() -> None:
    turns = {
        f"peer{i}": [
            Turn(tool_calls=(("write_scratchpad", {"content": f"obsolete {i}"}),)),
            Turn(
                tool_calls=(
                    ("write_scratchpad", {"content": f"latest {i}" * 30, "mode": "overwrite"}),
                    ("submit", {"answer": str(i + 3)}),
                )
            ),
        ]
        for i in range(3)
    }
    turns["finalizer0"] = [submit("5")]
    protocol = SwarmProtocol(
        SwarmConfig(
            n_agents=3,
            aggregation="finalizer",
            delivery=DeliverySpec(mode="push", push_max_chars=100),
        )
    )
    spec = swarm_spec(turns, protocol=protocol)
    episode, _ = await run_episode(spec)
    assert episode.ok and episode.outcome.final_answer == "5"
    assert episode.outcome.aggregation == "finalizer" and episode.outcome.votes is None
    assert episode.outcome.submissions == {"peer0": "3", "peer1": "4", "peer2": "5"}
    assert episode.grades["peer0"] == episode.grades["peer1"] == {"correct": 0.0}
    assert episode.grades["peer2"] == episode.grades["_system"] == {"correct": 1.0}
    assert episode.agents[-1].seat_key == ("finalizer", 0)
    (finalizer,) = [ctx for ctx in spec.policies["script"].calls if ctx.meta.role == "finalizer"]
    assert "You are finalizer0. Resolve the work of 3 peers" in finalizer.prompt_text
    first_message = finalizer.prompt_text.split("⟨user⟩", 1)[1]
    assert spec.task.prompt in first_message
    for i in range(3):
        assert f"[peer{i} scratchpad v2]\n" + f"latest {i}" * 30 in first_message
        assert f"[peer{i} submission]\n{i + 3}" in first_message
        assert f"obsolete {i}" not in first_message
    assert episode.calls[-1].tick == 2
    assert episode.calls[-1].segment_id == "finalizer0/g0"


def forced_answer(renderer: FakeRenderer, answer: str) -> list[int]:
    return renderer.encode_text('{"answer": "' + answer + '"}}') + [S["/call"], S["eot"]]


@pytest.mark.parametrize("on_exhaust", ["force_final", "none"])
async def test_max_ticks_finishes_every_active_peer(on_exhaust: str) -> None:
    limits = Limits(on_exhaust=on_exhaust)
    limits.episode.max_ticks = 1
    spec = swarm_spec({}, limits=limits)
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> list[int]:
        if ctx.meta.purpose == "final":
            return forced_answer(renderer, "5")
        return renderer.encode_completion(tool_calls=(("write_scratchpad", {"content": "work"}),))

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, _ = await run_episode(spec)
    assert episode.ok
    assert episode.limits_hit == {"_episode": ("episode.max_ticks",)}
    assert episode.outcome.submissions == {
        f"peer{i}": "5" if on_exhaust == "force_final" else None for i in range(3)
    }
    for i in range(3):
        calls = [call for call in episode.calls if call.agent_id == f"peer{i}"]
        assert len(calls) == (2 if on_exhaust == "force_final" else 1)
        if on_exhaust == "force_final":
            assert calls[-1].purpose == Purpose.FINAL and calls[-1].forced
            assert calls[-1].tick == 1


@pytest.mark.parametrize("on_exhaust", ["force_final", "none"])
async def test_consensus_stops_remaining_peers_using_verifier(on_exhaust: str) -> None:
    spec = swarm_spec(
        {},
        protocol=SwarmProtocol(SwarmConfig(n_agents=3, stop_on_consensus=2)),
        env=FractionEnv(),
        limits=Limits(on_exhaust=on_exhaust),
    )
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> list[int]:
        if ctx.meta.agent_id != "peer2":
            answer = "1/2" if ctx.meta.agent_id == "peer0" else "0.5"
            return renderer.encode_completion(tool_calls=(("submit", {"answer": answer}),))
        if ctx.meta.purpose == "final":
            return forced_answer(renderer, "3")
        return renderer.encode_completion(tool_calls=(("write_scratchpad", {"content": "work"}),))

    spec.policies["script"] = ScriptedPolicy("script", renderer, script, latency_s=0.001)
    episode, _ = await asyncio.wait_for(run_episode(spec), timeout=2)
    assert episode.ok and episode.outcome.final_answer == "1/2"
    done = next(
        event
        for event in episode.events
        if event.kind == EventKind.DONE and event.agent_id == "peer2"
    )
    assert done.data["ended_by"] == "consensus"
    calls = [call for call in episode.calls if call.agent_id == "peer2"]
    assert len(calls) <= 3
    assert episode.outcome.submissions["peer2"] == ("3" if on_exhaust == "force_final" else None)
    assert sum(call.forced for call in calls) == (on_exhaust == "force_final")


async def test_independent_has_only_submit_and_no_deliveries() -> None:
    protocol = IndependentProtocol(IndependentConfig(n_agents=3))
    spec = swarm_spec({f"peer{i}": [submit("5")] for i in range(3)}, protocol=protocol)
    episode, _ = await run_episode(spec)
    (role,) = protocol.roles()
    assert role.tools == ("submit",) and protocol.delivery.mode == "pull"
    assert episode.ok and episode.protocol == "independent"
    assert episode.outcome.votes == {"5": 3}
    assert all(call.reads == () for call in episode.calls)
    assert not episode.workspace_log
    for ctx in spec.policies["script"].calls:
        assert '"name": "submit"' in ctx.prompt_text
        assert "read_scratchpad" not in ctx.prompt_text
        assert "list_scratchpads" not in ctx.prompt_text
        assert "[workspace]" not in ctx.prompt_text


async def test_async_swarm_all_submit_and_are_not_replayable() -> None:
    spec = swarm_spec(shared_turns(), schedule="async")
    episode, _ = await run_episode(spec)
    assert episode.ok and not episode.replayable
    assert episode.outcome.submissions == {"peer0": "05", "peer1": "5", "peer2": "5"}
    assert all(call.tick is None for call in episode.calls)


async def test_lockstep_runs_are_identical_including_seeded_ties() -> None:
    turns = shared_turns()
    for index, peer in enumerate(turns):
        turns[peer][-1] = submit(str(index))
    left, right = swarm_spec(turns), swarm_spec(turns)
    left.clock, right.clock = FakeClock(), FakeClock()
    assert await run_episode(left) == await run_episode(right)


def test_roles_permissions_and_publish_flag() -> None:
    protocol = SwarmProtocol(
        SwarmConfig(n_agents=3, auto_publish="final_text", aggregation="finalizer")
    )
    peer, finalizer = protocol.roles()
    assert peer.count == 3 and peer.publish_final_text
    assert peer.permissions == Permissions(read_others=True, write_scratchpad=True)
    assert finalizer.count == 1 and not finalizer.publish_final_text
    assert finalizer.permissions == Permissions(read_others=True, write_scratchpad=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_agents": 0},
        {"n_agents": True},
        {"aggregation": "oracle_any"},
        {"finalizer": "unknown"},
        {"stop_on_consensus": -1},
        {"stop_on_consensus": 5},
        {"stop_on_consensus": True},
        {"auto_publish": "thinking"},
        {"peer_system_prompt": "Unknown {secret}"},
    ],
)
def test_invalid_swarm_configs_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        SwarmConfig(**overrides)


@pytest.mark.parametrize("overrides", [{"allow_resubmit": True}, {"finalizer": "peer0"}])
def test_deferred_options_fail_loudly(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError, match="not supported in v1"):
        SwarmConfig(**overrides)


@pytest.mark.parametrize(
    "name", ["swarm_n4", "swarm_n4_push", "swarm_n4_finalizer", "independent_n4"]
)
def test_yaml_entries_match_registered_config_types(name: str) -> None:
    path = Path(__file__).resolve().parents[1] / "src/marli/interact/configs" / f"{name}.yaml"
    data = yaml.safe_load(path.read_text())
    assert set(data) == {"protocol", "config", "notes"}
    cls = PROTOCOLS.get(data["protocol"])
    fields = data["config"]
    if "delivery" in fields:
        fields["delivery"] = DeliverySpec(**fields["delivery"])
    cfg = cls.config_type(**fields)
    protocol = cls(cfg)
    assert protocol.config.n_agents == 4
    assert protocol.name == data["protocol"]


def test_independent_rejects_accidental_sharing() -> None:
    with pytest.raises(ConfigError, match="independent"):
        replace(IndependentConfig(), delivery=DeliverySpec(mode="push"))


async def test_system_prompt_formats_peer_fields_and_preserves_literal_braces() -> None:
    template = "{agent_id}: {n_agents:02d} peers {peer_ids}; literal {{n_agents}}; \\boxed{{5}}"
    spec = swarm_spec(
        {"peer0": [submit("5")]},
        protocol=SwarmProtocol(SwarmConfig(n_agents=1, peer_system_prompt=template)),
    )
    episode, _ = await run_episode(spec)
    assert episode.ok
    assert "peer0: 01 peers peer0; literal {n_agents}; \\boxed{5}" in (
        spec.policies["script"].calls[0].prompt_text
    )


class BoxedArithEnv(ArithEnv):
    def canonical(self, submission: str | None) -> str | None:
        import re

        if submission is None:
            return None
        matches = re.findall(r"\\boxed\{([^{}]*)\}", submission)
        return super().canonical(matches[-1] if matches else submission)


class NoToolDebateRenderer(FakeRenderer):
    def forced_tool_prefix(self, tool_name: str) -> list[int]:
        pytest.fail(f"Debate cannot force an unadvertised tool: {tool_name}")


async def test_debate_two_rounds_push_replies_keep_latest_answers_and_vote_deterministically() -> (
    None
):
    from marli.interact.configs import build_protocol
    from marli.interact.protocols.presets import DebateConfig, DebateProtocol

    turns = {
        f"peer{i}": [
            Turn(f"Round one from {i}: candidate \\boxed{{{i}}}.", thinking=f"private {i}"),
            Turn(f"Round two from {i}: checked \\boxed{{{('05', '5', '6')[i]}}}."),
        ]
        for i in range(3)
    }
    protocol = build_protocol("debate_n3_r2")
    assert isinstance(protocol, DebateProtocol) and isinstance(protocol.config, DebateConfig)
    spec = swarm_spec(turns, protocol=protocol, env=BoxedArithEnv())
    spec.renderers["script"] = NoToolDebateRenderer
    spec.clock = FakeClock()
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.replayable
    assert episode.outcome.submissions == {
        peer: replies[1].content for peer, replies in turns.items()
    }
    assert episode.outcome.final_answer == turns["peer0"][1].content
    assert episode.outcome.votes == {"5": 2, "6": 1}
    assert episode.grades["_system"] == {"correct": 1.0}
    assert len(episode.calls) == len(episode.workspace_log) == 6
    assert all(call.purpose == Purpose.ACT and not call.forced for call in episode.calls)
    assert {e.data["ended_by"] for e in episode.events if e.kind == EventKind.DONE} == {"max_ticks"}
    for i in range(3):
        prompts = [ctx for ctx in spec.policies["script"].calls if ctx.meta.agent_id == f"peer{i}"]
        assert len(prompts) == 2
        assert "over 2 rounds" in prompts[0].prompt_text and r"\boxed{}" in prompts[0].prompt_text
        assert "⟨tools⟩" not in prompts[0].prompt_text
        for j in range(3):
            if i != j:
                assert turns[f"peer{j}"][0].content in prompts[1].prompt_text
                assert f"private {j}" not in prompts[1].prompt_text
        calls = [call for call in episode.calls if call.agent_id == f"peer{i}"]
        assert [call.tick for call in calls] == [0, 1]
        assert {(r.writer, r.version, r.via) for r in calls[1].reads} == {
            (f"peer{j}", 1, ReadVia.PUSH) for j in range(3) if i != j
        }
        for call, ctx in zip(calls, prompts, strict=True):
            assert tuple(buffers[call.segment_id][: call.prompt_len]) == ctx.prompt_ids
    other = swarm_spec(turns, protocol=build_protocol("debate_n3_r2"), env=BoxedArithEnv())
    other.renderers["script"] = NoToolDebateRenderer
    other.clock = FakeClock()
    assert await run_episode(other) == (episode, buffers)
    assert spec.limits.on_no_tool_call == "nudge" and spec.limits.episode.max_ticks == 64


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_debate_rejects_invalid_round_count(rounds: int) -> None:
    from marli.interact.protocols.presets import DebateConfig

    with pytest.raises(ConfigError, match="rounds"):
        DebateConfig(rounds=rounds)


@pytest.mark.parametrize(
    "protocol_name", ["single", "swarm", "independent", "coordinator", "debate"]
)
def test_non_session_protocols_pin_one_session_and_warn(protocol_name: str) -> None:
    from marli.interact.configs import build_protocol
    from marli.interact.limits import SessionLimits

    protocol = build_protocol(protocol_name)
    limits = Limits(session=SessionLimits(max_sessions=3))
    with pytest.warns(
        UserWarning, match="sessions are owned by the multi_session protocol"
    ) as caught:
        adjusted = protocol.adjust_limits(limits)
    assert len(caught) == 1
    assert adjusted.session.max_sessions == 1 and limits.session.max_sessions == 3
