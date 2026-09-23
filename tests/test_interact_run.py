"""Episode integration validates before setup and always cleans up agent tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace

import pytest
from _marli_test_envs import ArithEnv

from marli.errors import ConfigError
from marli.interact.agent import SamplingOverrides
from marli.interact.limits import Limits
from marli.interact.protocols.single import SingleConfig, SingleProtocol
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.scheduler import FakeClock
from marli.interact.system import RoleSpec, SystemIO, get_protocol
from marli.interact.types import Outcome, Purpose, SegmentStart, Termination, Usage
from marli.policy.base import CallMeta, ChatReply
from marli.policy.scripted import ScriptedChatPolicy, ScriptedPolicy, Turn, turns_by_agent
from marli.render.base import Msg, ParsedToolCall
from marli.render.fake import FakeRenderer
from marli.seeds import cache_salt, derive_seed


def single_spec() -> EpisodeSpec:
    renderer = FakeRenderer()
    env = ArithEnv()
    script = turns_by_agent(
        renderer, {"solver0": [Turn(tool_calls=(("submit", {"answer": " 05 "}),))]}
    )
    policy = ScriptedPolicy("script", renderer, script)
    policy.deterministic = True
    return EpisodeSpec(
        SingleProtocol(),
        env,
        env.task,
        {"solver": "script"},
        {"script": policy},
        {"script": FakeRenderer},
        Limits(),
        run_seed=42,
        episode_idx=2,
        config_hash="abcdefgh1234",
        backend="scripted",
    )


async def test_single_end_to_end_fields_grades_and_exact_buffer() -> None:
    spec = single_spec()
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.replayable and not episode.errors
    assert (
        episode.group_id == "arithmetic/abcdefgh" and episode.episode_id == "arithmetic/abcdefgh/e2"
    )
    assert episode.episode_idx == 2 and episode.task_id == spec.task.task_id
    assert episode.protocol == "single" and episode.backend == "scripted"
    assert episode.config_hash == spec.config_hash
    assert episode.outcome.final_answer == " 05 "
    assert episode.outcome.submissions == {"solver0": " 05 "}
    assert episode.outcome.aggregation == "single"
    assert episode.grades == {"solver0": {"correct": 1.0}, "_system": {"correct": 1.0}}
    assert spec.env.graded == [" 05 ", " 05 "]
    assert spec.env.setup_count == spec.env.teardown_count == 1
    assert episode.agents[0].seat_key == ("solver", 0)
    assert episode.segments[0].start_reason == SegmentStart.START
    (call,) = episode.calls
    assert call.call_id == "arithmetic/abcdefgh/e2/solver0/c0"
    assert call.purpose == Purpose.ACT and call.tick == 0 and not call.forced
    assert call.seed == derive_seed(42, "arithmetic", 2, "solver0", 0)
    (ctx,) = spec.policies["script"].calls
    assert buffers[call.segment_id] == list(ctx.prompt_ids + call.completion_ids)
    assert episode.metrics["total_gen"] == len(call.completion_ids)
    assert episode.metrics["total_prompt"] == call.prompt_len
    assert episode.metrics["cp_calls"] == 1


async def test_chat_single_records_usage_and_unique_cache_salt() -> None:
    spec = single_spec()

    def reply(messages: Sequence[Msg], meta: CallMeta) -> ChatReply:
        if meta.call_index == 0:
            return ChatReply(
                "Let me think.", None, (), Termination.STOP, Usage(10, 4, tokenizer="api:test")
            )
        return ChatReply(
            "Done",
            None,
            (
                ParsedToolCall(
                    "submit",
                    {"answer": "5"},
                    '{"answer": "5"}',
                    True,
                    "call0",
                ),
            ),
            Termination.STOP,
            Usage(20, 6, tokenizer="api:test"),
        )

    policy = ScriptedChatPolicy("chat", reply)
    policy.deterministic = True
    spec.seating = {"solver": "chat"}
    spec.policies = {"chat": policy}
    spec.renderers = {}
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.replayable and not buffers and not episode.segments
    assert episode.outcome.final_answer == "5"
    # API seats are accounted separately from token-level compute
    assert episode.metrics["total_gen"] == 0 and episode.metrics["total_prompt"] == 0
    assert episode.metrics["api_calls"] == len(episode.calls)
    assert episode.metrics["api_gen_tokens"] == 10
    for index, call in enumerate(episode.calls):
        assert call.segment_id == "" and call.prompt_len == 0
        assert call.completion_ids == () and call.logprobs is None
        assert call.text == ("Let me think." if index == 0 else "Done")
        assert call.usage.tokenizer == "api:test"
        assert policy.calls[index].cache_salt == cache_salt(42, "arithmetic", 2, "solver0", index)
    assert policy.calls[0].cache_salt != policy.calls[1].cache_salt
    assert [message.role for message in policy.calls[1].messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [tool.name for tool in policy.calls[0].tools] == ["submit"]


@pytest.mark.parametrize(
    "problem",
    [
        "unseated",
        "missing_policy",
        "missing_renderer",
        "renderer_mismatch",
        "trainable_chat",
        "temperature",
        "top_p",
        "top_k",
        "context",
        "tool",
        "schedule",
        "context_kind",
    ],
)
async def test_invalid_specs_fail_before_setup_and_sampling(problem: str) -> None:
    spec = single_spec()
    if problem == "unseated":
        spec.seating = {}
    elif problem == "missing_policy":
        spec.policies = {}
    elif problem == "missing_renderer":
        spec.renderers = {}
    elif problem == "renderer_mismatch":
        spec.policies["script"].renderer_name = "different"
    elif problem == "trainable_chat":
        policy = ScriptedChatPolicy("script", lambda messages, meta: None)
        policy.trainable = True
        spec.policies["script"] = policy
    elif problem in {"temperature", "top_p", "top_k"}:
        spec.policies["script"].trainable = True
        spec.sampling = {"solver": SamplingOverrides(**{problem: 3 if problem == "top_k" else 0.7})}
    elif problem == "context":
        spec.policies["script"].max_seq_len = 100
    elif problem == "tool":
        spec.protocol = SingleProtocol(SingleConfig(env_tools=("missing",)))
    elif problem == "schedule":
        spec.schedule = "invalid"
    else:

        class BadContextProtocol(SingleProtocol):
            def roles(self) -> list[RoleSpec]:
                (role,) = super().roles()
                return [replace(role, context=replace(role.context, kind="invalid"))]

        spec.protocol = BadContextProtocol()
    with pytest.raises(ConfigError):
        await run_episode(spec)
    assert spec.env.setup_count == spec.env.teardown_count == 0
    for policy in spec.policies.values():
        assert policy.calls == []


@pytest.mark.parametrize("schedule", ["lockstep", "async"])
async def test_wall_clock_timeout_cancels_agents_and_tears_down(schedule: str) -> None:
    spec = single_spec()
    spec.schedule = schedule
    clock = FakeClock()
    spec.clock = clock
    spec.limits.episode.max_wall_s = 2
    policy = spec.policies["script"]
    spec.policies["script"] = ScriptedPolicy(
        "script",
        FakeRenderer(),
        policy._script,
        clock=clock,
        latency_s=10,
    )
    task = asyncio.create_task(run_episode(spec))
    for _ in range(20):
        await asyncio.sleep(0)
        if spec.policies["script"].calls:
            break
    assert len(spec.policies["script"].calls) == 1
    clock.advance(2)
    episode, _ = await asyncio.wait_for(task, timeout=1)
    assert not episode.ok and episode.errors == ("episode wall-clock limit",)
    assert episode.limits_hit == {"_episode": ("episode.max_wall_s",)}
    assert spec.env.teardown_count == 1
    assert not clock._sleepers
    assert len(episode.calls) == 1
    assert episode.calls[0].termination == Termination.ERROR


async def test_setup_failure_still_tears_down() -> None:
    class BrokenEnv(ArithEnv):
        async def setup(self) -> None:
            raise TypeError("setup broken")

    spec = single_spec()
    spec.env = BrokenEnv()
    with pytest.raises(TypeError, match="setup broken"):
        await run_episode(spec)
    assert spec.env.teardown_count == 1


def test_single_registry_config() -> None:
    config = SingleConfig("custom prompt", ("write_scratchpad",))
    protocol = get_protocol("single", config)
    (role,) = protocol.roles()
    assert protocol.config is config
    assert role.tools == ("submit", "write_scratchpad")
    assert role.system_prompt == "custom prompt"


async def test_explicit_determinism_flag_is_respected() -> None:
    spec = single_spec()
    spec.policies["script"].deterministic = False
    episode, _ = await run_episode(spec)
    assert not episode.replayable


async def test_finalize_and_canonicalize_use_environment_and_new_agent() -> None:
    class FinalizerProtocol(SingleProtocol):
        def roles(self) -> list[RoleSpec]:
            (solver,) = super().roles()
            return [solver, replace(solver, role="finalizer")]

        async def run(self, io: SystemIO) -> Outcome:
            await super().run(io)
            result = await io.finalize(
                "finalizer", agent_id="finalizer0", first_message="Finalize."
            )
            assert io.canonicalize(result.submission) == "5"
            return Outcome(result.submission, {result.agent_id: result.submission}, "finalizer")

    spec = single_spec()
    spec.protocol = FinalizerProtocol()
    spec.seating["finalizer"] = "script"
    renderer = FakeRenderer()
    spec.policies["script"] = ScriptedPolicy(
        "script",
        renderer,
        turns_by_agent(
            renderer,
            {
                agent: [Turn(tool_calls=(("submit", {"answer": " 05 "}),))]
                for agent in ("solver0", "finalizer0")
            },
        ),
    )
    episode, _ = await run_episode(spec)
    assert [agent.agent_id for agent in episode.agents] == ["solver0", "finalizer0"]
    assert episode.grades == {
        agent: {"correct": 1.0} for agent in ("solver0", "finalizer0", "_system")
    }
    assert episode.calls[-1].tick == 1


def test_invalid_delivery_is_rejected_at_construction() -> None:
    spec = single_spec()
    with pytest.raises(ConfigError, match="delivery mode"):
        replace(spec.delivery, mode="invalid")


@pytest.mark.parametrize("peer_answer", ["4", None])
async def test_outcome_submissions_override_runtime_for_individual_grades(
    peer_answer: str | None,
) -> None:
    class OutcomeProtocol(SingleProtocol):
        async def run(self, io: SystemIO) -> Outcome:
            result = await super().run(io)
            return replace(result, submissions={"solver0": peer_answer})

    spec = single_spec()
    spec.protocol = OutcomeProtocol()
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == " 05 "
    assert episode.grades == {"solver0": {"correct": 0.0}, "_system": {"correct": 1.0}}
    assert spec.env.graded == [peer_answer, " 05 "]


async def test_runtime_submission_grades_fall_back_for_absent_outcome_agents() -> None:
    class OutcomeProtocol(SingleProtocol):
        async def run(self, io: SystemIO) -> Outcome:
            result = await super().run(io)
            return replace(result, submissions={})

    spec = single_spec()
    spec.protocol = OutcomeProtocol()
    episode, _ = await run_episode(spec)
    assert episode.grades["solver0"] == {"correct": 1.0}


@pytest.mark.parametrize("override", [True, False])
async def test_protocol_delivery_override_or_episode_fallback(override: bool) -> None:
    from marli.interact.workspace import DeliverySpec

    class DeliveryProtocol(SingleProtocol):
        delivery = DeliverySpec(mode="pull") if override else None

        async def run(self, io: SystemIO) -> Outcome:
            assert io.workspace.delivery.mode == ("pull" if override else "push")
            return await super().run(io)

    spec = single_spec()
    spec.protocol = DeliveryProtocol()
    spec.delivery = DeliverySpec(mode="push")
    episode, _ = await run_episode(spec)
    assert episode.ok


async def test_vote_ties_use_episode_rng_and_none_is_excluded() -> None:
    class VoteProtocol(SingleProtocol):
        async def run(self, io: SystemIO) -> Outcome:
            state = io.rng.getstate()
            expected = io.rng.choice([" 05 ", "6"])
            io.rng.setstate(state)
            answer, votes = await io.vote({"a": None, "b": " 05 ", "c": "6", "d": None})
            assert answer == expected
            assert votes == {"5": 1, "6": 1}
            assert await io.vote({"a": None, "b": None}) == (None, {})
            assert await io.vote({}) == (None, {})
            return Outcome(answer, {}, "vote", votes)

    spec = single_spec()
    spec.protocol = VoteProtocol()
    episode, _ = await run_episode(spec)
    assert episode.ok and not episode.calls


async def test_vote_compares_only_first_cluster_members_and_raw_key_fallback() -> None:
    class NearEnv(ArithEnv):
        def __init__(self) -> None:
            super().__init__()
            self.pairs: list[tuple[str | None, str | None]] = []

        def canonical(self, submission: str | None) -> str | None:
            return None

        async def same_answer(self, a: str | None, b: str | None) -> bool:
            self.pairs.append((a, b))
            return a is not None and b is not None and abs(int(a) - int(b)) <= 1

    class VoteProtocol(SingleProtocol):
        async def run(self, io: SystemIO) -> Outcome:
            answer, votes = await io.vote({"a": "1", "b": "2", "c": "3", "d": None})
            return Outcome(answer, {}, "vote", votes)

    spec = single_spec()
    spec.protocol = VoteProtocol()
    env = NearEnv()
    spec.env = env
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "1"
    assert episode.outcome.votes == {"1": 2, "3": 1}
    assert env.pairs == [("2", "1"), ("3", "1")]


@pytest.mark.parametrize(
    "prompt", ['Answer as {"answer": ...}.', "{unknown}", "{", "{n_agents:bad}"]
)
async def test_system_prompt_format_validation_precedes_env_setup(prompt: str) -> None:
    spec = single_spec()
    spec.protocol = SingleProtocol(SingleConfig(system_prompt=prompt))
    with pytest.raises(ConfigError, match="system_prompt"):
        await run_episode(spec)
    assert spec.env.setup_count == spec.env.teardown_count == 0
    assert not spec.policies["script"].calls


async def test_system_prompt_n_agents_is_role_count() -> None:
    class TwoRoles(SingleProtocol):
        def roles(self) -> list[RoleSpec]:
            (solver,) = super().roles()
            return [
                replace(solver, count=3, system_prompt="Solvers: {n_agents}."),
                replace(solver, role="idle", count=7),
            ]

    spec = single_spec()
    spec.protocol = TwoRoles()
    spec.seating["idle"] = "script"
    episode, buffers = await run_episode(spec)
    (call,) = episode.calls
    (ctx,) = spec.policies["script"].calls
    assert "Solvers: 3." in ctx.prompt_text
    assert buffers[call.segment_id] == list(ctx.prompt_ids + call.completion_ids)


@pytest.mark.parametrize("deterministic", [None, False, True])
async def test_determinism_is_duck_typed_and_only_seated_policies_are_checked(
    deterministic: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = single_spec()
    seated = spec.policies["script"]
    if deterministic is None:  # a policy without the attribute is not replayable
        del seated.deterministic
        monkeypatch.delattr(type(seated), "deterministic")
    else:
        seated.deterministic = deterministic
    unused = ScriptedChatPolicy("unused", lambda messages, meta: None)
    unused.trainable = True
    unused.max_seq_len = 1
    unused.deterministic = False
    spec.policies["unused"] = unused
    episode, buffers = await run_episode(spec)
    assert episode.replayable is (deterministic is True)
    assert not unused.calls
    (call,) = episode.calls
    assert buffers[call.segment_id] == list(seated.calls[0].prompt_ids + call.completion_ids)
