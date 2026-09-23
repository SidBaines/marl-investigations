"""Exercise session ablations through the real runtime and its token records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from _marli_test_envs import ArithEnv

from marli.config import compose
from marli.errors import ConfigError
from marli.interact.configs import resolve_protocol
from marli.interact.limits import CallLimits, Limits, SessionLimits
from marli.interact.protocols.multi_session import MultiSessionConfig, MultiSessionProtocol
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.scheduler import FakeClock
from marli.interact.system import get_protocol
from marli.interact.types import Purpose, SegmentStart, record_to_dict
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, turns_by_agent
from marli.render.fake import SPECIAL_IDS as S
from marli.render.fake import FakeRenderer

CARRY_MODES = ("compaction", "notes", "both", "tail")
SUMMARY = "The intermediate result is five; check the addition."
NOTES = "Saved candidate: 5."
PRIVATE_WORK = "Unsaved first-session reasoning."
CONFIGS = Path(__file__).resolve().parents[1] / "src/marli/interact/configs"


def make_spec(
    config: MultiSessionConfig, turns: list[Turn], *, limits: Limits | None = None
) -> EpisodeSpec:
    env = ArithEnv()
    renderer = FakeRenderer()
    return EpisodeSpec(
        protocol=MultiSessionProtocol(config),
        env=env,
        task=env.task,
        seating={"solver": "script"},
        policies={
            "script": ScriptedPolicy(
                "script", renderer, turns_by_agent(renderer, {"solver0": turns})
            )
        },
        renderers={"script": FakeRenderer},
        limits=limits or Limits(session=SessionLimits(max_sessions=config.sessions)),
        run_seed=17,
        clock=FakeClock(),
    )


def ending_turn(carry: str, budget_end: bool) -> Turn:
    return Turn(
        content=PRIVATE_WORK + "x" * 128,
        tool_calls=((("write_notes", {"content": NOTES}),) if carry in ("notes", "both") else ())
        + (() if budget_end else (("end_session", {}),)),
    )


def session_spec(carry: str, budget_end: bool, *, sessions: int = 2) -> EpisodeSpec:
    first = ending_turn(carry, budget_end)
    renderer = FakeRenderer()
    first_tokens = len(renderer.encode_completion(first.content, tool_calls=first.tool_calls))
    reserve = 128 if carry in ("compaction", "both") else 0
    config = MultiSessionConfig(
        sessions=sessions,
        carry=carry,
        tail_tokens=24,
        session_tokens=first_tokens + reserve if budget_end else 4096,
        tools=("submit", "end_session", "read_scratchpad")
        if carry == "tail" and not budget_end
        else ("submit", "end_session"),
    )
    limits = Limits(
        call=CallLimits(min_call_tokens=1),
        session=SessionLimits(
            max_sessions=sessions,
            max_gen_tokens=first_tokens + 128 if budget_end else 4096,
            carry_reserve=128,
            carry_max_tokens=128,
        ),
        max_nudges=sessions + 1,
    )
    turns = [first]
    if carry in ("compaction", "both"):
        turns.append(Turn(SUMMARY))
    turns.append(Turn(tool_calls=(("submit", {"answer": "5"}),)))
    return make_spec(config, turns, limits=limits)


@pytest.mark.parametrize("carry", CARRY_MODES)
@pytest.mark.parametrize("budget_end", [False, True], ids=["end_session", "token_budget"])
async def test_session_carry_preserves_only_selected_state(carry: str, budget_end: bool) -> None:
    spec = session_spec(carry, budget_end)
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.replayable and not episode.errors
    assert episode.outcome.final_answer == "5"
    assert episode.outcome.submissions == {"solver0": "5"}
    assert episode.outcome.aggregation == "multi_session"
    assert episode.grades == {"solver0": {"correct": 1.0}, "_system": {"correct": 1.0}}
    assert episode.metrics["n_sessions"] == 2
    assert episode.metrics["n_agents"] == 1
    assert episode.metrics["tool_errors"] == 0
    assert spec.env.graded == ["5", "5"]
    assert spec.env.setup_count == spec.env.teardown_count == 1
    (agent,) = episode.agents
    assert (agent.agent_id, agent.role, agent.seat_key) == ("solver0", "solver", ("solver", 0))
    first, second = episode.segments
    # The runtime labels the initial context START and subsequent sessions SESSION.
    assert first.start_reason == SegmentStart.START and first.carry_from is None
    assert second.start_reason == SegmentStart.SESSION and second.carry_from == first.segment_id
    assert (first.session_idx, second.session_idx) == (0, 1)

    policy = spec.policies["script"]
    renderer = FakeRenderer()
    first_prompt = policy.calls[0].prompt_ids
    user_start = first_prompt.index(S["user"]) + 1
    assert renderer.decode(first_prompt[user_start : first_prompt.index(S["eot"], user_start)]) == (
        "[Session 1 of 2]\n\n" + spec.env.task_message("solver")
    )
    next_call = next(call for call in episode.calls if call.session_idx == 1)
    next_prompt = buffers[second.segment_id][: next_call.prompt_len]
    user_start = next_prompt.index(S["user"]) + 1
    user_end = next_prompt.index(S["eot"], user_start)
    next_message = renderer.decode(next_prompt[user_start:user_end])
    assert next_message.startswith(
        "[Session 2 of 2] — final session: you must submit before it ends\n\n"
        + spec.env.task_message("solver")
    )
    assert (SUMMARY in next_message) == (carry in ("compaction", "both"))
    assert (NOTES in next_message) == (carry in ("notes", "both"))
    assert PRIVATE_WORK not in renderer.decode(next_prompt)
    if carry == "tail":
        assert "[Tail of your previous session]\n" + "x" * 24 in next_message
    assert next_prompt[user_end + 1 :] == [S["asst"]]

    carries = [call for call in episode.calls if call.purpose == Purpose.CARRY]
    assert len(carries) == int(carry in ("compaction", "both"))
    if carries:
        (call,) = carries
        assert call.forced == budget_end
        assert call.session_idx == 0 and call.segment_id == first.segment_id
        assert renderer.parse(call.completion_ids, []).content == SUMMARY
    assert not any(call.purpose == Purpose.COMPACT for call in episode.calls)
    hits = episode.limits_hit.get("solver0", ())
    assert ("session.max_gen_tokens" in hits) == budget_end
    if budget_end:
        assert len(episode.calls[0].completion_ids) == (
            spec.protocol.config.session_tokens
            - (spec.limits.session.carry_reserve if carry in ("compaction", "both") else 0)
        )
    for call, actual in zip(episode.calls, policy.calls, strict=True):
        buffer = buffers[call.segment_id]
        assert tuple(buffer[: call.prompt_len]) == actual.prompt_ids
        assert tuple(buffer[call.prompt_len : call.prompt_len + len(call.completion_ids)]) == (
            call.completion_ids
        )


@pytest.mark.parametrize("carry", CARRY_MODES)
@pytest.mark.parametrize("budget_end", [False, True])
@pytest.mark.parametrize("sessions", [2, 3])
async def test_last_session_forces_submission(carry: str, budget_end: bool, sessions: int) -> None:
    spec = session_spec(carry, budget_end, sessions=sessions)
    renderer = FakeRenderer()
    first = ending_turn(carry, budget_end)

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose == "carry":
            return renderer.encode_completion(SUMMARY)
        if ctx.meta.purpose == "final":
            return [*renderer.encode_text('{"answer": "5"}}'), S["/call"], S["eot"]]
        return renderer.encode_completion(first.content, tool_calls=first.tool_calls)

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.outcome.final_answer == "5"
    assert episode.metrics["n_sessions"] == sessions
    assert len(episode.segments) == sessions
    assert [segment.session_idx for segment in episode.segments] == list(range(sessions))
    for previous, current in zip(episode.segments[:-1], episode.segments[1:], strict=True):
        assert current.start_reason == SegmentStart.SESSION
        assert current.carry_from == previous.segment_id
    assert "session.max_sessions" in episode.limits_hit["solver0"]
    final = episode.calls[-1]
    assert final.purpose == Purpose.FINAL and final.forced
    assert final.session_idx == sessions - 1
    assert final.tool_calls[0].name == "submit" and final.tool_calls[0].error is None
    prefix = renderer.forced_tool_prefix("submit")
    assert buffers[final.segment_id][final.prompt_len - len(prefix) : final.prompt_len] == prefix
    assert len([call for call in episode.calls if call.purpose == Purpose.CARRY]) == (
        sessions - 1 if carry in ("compaction", "both") else 0
    )


@pytest.mark.parametrize("carry", CARRY_MODES)
async def test_submit_ends_episode_in_first_session(carry: str) -> None:
    spec = make_spec(
        MultiSessionConfig(carry=carry), [Turn(tool_calls=(("submit", {"answer": "5"}),))]
    )
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "5"
    assert episode.metrics["n_sessions"] == 1
    assert len(episode.segments) == len(episode.calls) == 1
    assert episode.calls[0].purpose == Purpose.ACT and not episode.calls[0].forced


@pytest.mark.parametrize("carry", CARRY_MODES)
@pytest.mark.parametrize("budget_end", [False, True])
async def test_scripted_sessions_replay_identically(carry: str, budget_end: bool) -> None:
    first = await run_episode(session_spec(carry, budget_end))
    second = await run_episode(session_spec(carry, budget_end))
    assert first == second
    assert json.dumps(record_to_dict(first[0]), sort_keys=True) == json.dumps(
        record_to_dict(second[0]), sort_keys=True
    )


@pytest.mark.parametrize("carry", CARRY_MODES)
def test_role_config_and_registered_factory(carry: str) -> None:
    config = MultiSessionConfig(carry=carry, env_tools=("write_scratchpad", "submit"))
    protocol = get_protocol("multi_session", config)
    (role,) = protocol.roles()
    assert protocol.config is config
    assert role.role == "solver" and role.count == 1
    expected_notes = ("read_notes", "write_notes") if carry in ("notes", "both") else ()
    end_tools = () if carry == "tail" else ("end_session",)
    assert role.tools == ("submit", *end_tools, *expected_notes, "write_scratchpad")
    assert role.context.kind == carry
    assert role.context.compact_threshold == 0
    assert role.context.tail_tokens == (2048 if carry == "tail" else 0)
    assert role.context.notes_cap_chars == 4000
    assert "up to 3 sessions" in role.system_prompt
    assert "session token budget" in role.system_prompt
    assert "context is cleared" in role.system_prompt


async def test_custom_prompt_tools_and_environment_tools() -> None:
    config = MultiSessionConfig(
        sessions=2,
        carry="notes",
        tools=("submit", "end_session", "read_notes"),
        env_tools=("write_scratchpad",),
        system_prompt="{agent_id} ({role}/{n_agents}): {sessions} sessions, {carry} carry.",
    )
    spec = make_spec(
        config,
        [
            Turn(tool_calls=(("write_scratchpad", {"content": "environment action"}),)),
            Turn(tool_calls=(("submit", {"answer": "5"}),)),
        ],
    )
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "5" and episode.metrics["tool_errors"] == 0
    assert "solver0 (solver/1): 2 sessions, notes carry." in (
        spec.policies["script"].calls[0].prompt_text
    )


@pytest.mark.parametrize("carry", ["compaction", "both"])
def test_optional_compaction_threshold(carry: str) -> None:
    (role,) = MultiSessionProtocol(MultiSessionConfig(carry=carry, compact_threshold=8192)).roles()
    assert role.context.compact_threshold == 8192


@pytest.mark.parametrize("carry", ["notes", "both"])
async def test_notes_cap_is_applied_to_workspace_and_carry(carry: str) -> None:
    turns = [
        Turn(
            tool_calls=(
                ("write_notes", {"content": NOTES}),
                ("write_notes", {"content": NOTES[-10:]}),
                ("end_session", {}),
            )
        )
    ]
    if carry == "both":
        turns.append(Turn(SUMMARY))
    turns.append(Turn(tool_calls=(("submit", {"answer": "5"}),)))
    spec = make_spec(MultiSessionConfig(carry=carry, notes_cap_chars=10), turns)
    episode, _ = await run_episode(spec)
    assert "cap of 10 chars" in episode.calls[0].tool_calls[0].error
    (write,) = episode.workspace_log
    assert write.key == "notes" and write.content == NOTES[-10:]
    assert f"[Your notes]\n{NOTES[-10:]}" in spec.policies["script"].calls[-1].prompt_text


async def test_protocol_sets_session_count_without_mutating_caller_limits() -> None:
    spec = make_spec(
        MultiSessionConfig(sessions=2),
        [
            Turn(tool_calls=(("end_session", {}),)),
            Turn(SUMMARY),
            Turn(tool_calls=(("submit", {"answer": "5"}),)),
        ],
    )
    spec.limits.session.max_sessions = 7
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "5" and episode.metrics["n_sessions"] == 2
    assert spec.limits.session.max_sessions == 7


@pytest.mark.parametrize(
    "overrides",
    [
        {"carry": "invalid"},
        {"sessions": 0},
        {"sessions": 1},
        {"session_tokens": 0},
        {"session_tokens": True},
        {"session_tokens": 1.5},
        {"sessions": -1},
        {"sessions": True},
        {"sessions": 1.5},
        {"tail_tokens": 0},
        {"notes_cap_chars": 0},
        {"compact_threshold": -1},
        {"compact_threshold": True},
        {"carry": "tail", "compact_threshold": 1},
        {"carry": "notes", "compact_threshold": 1},
    ],
)
def test_invalid_config(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        MultiSessionConfig(**overrides)


def test_unknown_config_keys_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown"):
        compose(MultiSessionConfig, overrides=["unknown=true"])


@pytest.mark.parametrize("carry", CARRY_MODES)
async def test_three_session_presets_run(carry: str) -> None:
    config = resolve_protocol(f"multi_session_s3_{carry}")[1]
    assert config.sessions == 3 and config.carry == carry
    if carry == "tail":
        config = replace(config, tools=("submit", "end_session", "read_scratchpad"))
    turns: list[Turn] = []
    for _ in range(2):
        turns.append(Turn(tool_calls=(("end_session", {}),)))
        if carry in ("compaction", "both"):
            turns.append(Turn(SUMMARY))
    turns.append(Turn(tool_calls=(("submit", {"answer": "5"}),)))
    episode, _ = await run_episode(make_spec(config, turns))
    assert episode.ok and episode.outcome.final_answer == "5"
    assert episode.metrics["n_sessions"] == 3


async def test_self_refine_preset_carries_candidate_for_second_session() -> None:
    config = resolve_protocol("self_refine_s2")[1]
    assert config.sessions == 2 and config.carry == "notes"
    spec = make_spec(
        config,
        [
            Turn(tool_calls=(("write_notes", {"content": "Candidate: 4."}), ("end_session", {}))),
            Turn(tool_calls=(("submit", {"answer": "5"}),)),
        ],
    )
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "5" and episode.metrics["n_sessions"] == 2
    prompt = spec.policies["script"].calls[-1].prompt_text
    assert "[Your notes]\nCandidate: 4." in prompt
    assert "In the second session, check" in prompt and "refine" in prompt


@pytest.mark.parametrize("carry", CARRY_MODES)
def test_session_budgets_divide_available_compute_without_mutating_limits(carry: str) -> None:
    limits = Limits()
    adjusted = MultiSessionProtocol(MultiSessionConfig(carry=carry)).adjust_limits(limits)
    assert adjusted.session.max_sessions == 3
    assert (
        adjusted.session.max_gen_tokens
        == (limits.agent.max_gen_tokens - limits.agent.final_reserve) // 3
    )
    assert adjusted.session.carry_reserve == (
        0 if carry in ("notes", "tail") else limits.session.carry_reserve
    )
    assert limits.session == SessionLimits()
    explicit = MultiSessionProtocol(MultiSessionConfig(carry=carry, session_tokens=3000))
    assert explicit.adjust_limits(limits).session.max_gen_tokens == 3000
    with pytest.raises(ConfigError, match="session_tokens"):
        MultiSessionProtocol(MultiSessionConfig(session_tokens=20000)).adjust_limits(limits)


@pytest.mark.parametrize("carry", CARRY_MODES)
async def test_default_s3_budgets_reach_all_sessions_then_use_final_reserve(carry: str) -> None:
    config = resolve_protocol(f"multi_session_s3_{carry}")[1]
    spec = make_spec(config, [], limits=Limits(max_nudges=1000))
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose == "final":
            return [*renderer.encode_text('{"answer": "5"}}'), S["/call"], S["eot"]]
        return renderer.encode_completion("x" * (ctx.spec.max_tokens - 1))

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, _ = await run_episode(spec)
    assert episode.ok and episode.outcome.final_answer == "5"
    assert episode.metrics["n_sessions"] == 3
    per_session = (spec.limits.agent.max_gen_tokens - spec.limits.agent.final_reserve) // 3
    for session in range(3):
        calls = [
            call
            for call in episode.calls
            if call.session_idx == session and call.purpose != Purpose.FINAL
        ]
        assert calls and sum(len(call.completion_ids) for call in calls) <= per_session
        first = next(
            ctx
            for call, ctx in zip(episode.calls, spec.policies["script"].calls, strict=True)
            if call.session_idx == session
        )
        assert f"{per_session} generated tokens" in first.prompt_text
        assert f"⟨user⟩[Session {session + 1} of 3]" in first.prompt_text
    assert episode.calls[-1].purpose == Purpose.FINAL
    assert episode.metrics["total_gen"] <= spec.limits.agent.max_gen_tokens


async def test_notes_persist_and_append_across_three_sessions() -> None:
    spec = make_spec(
        MultiSessionConfig(carry="notes"),
        [
            Turn(tool_calls=(("write_notes", {"content": "tried A"}), ("end_session", {}))),
            Turn(tool_calls=(("read_notes", {}),)),
            Turn(
                tool_calls=(
                    ("write_notes", {"content": "tried B", "mode": "append"}),
                    ("end_session", {}),
                )
            ),
            Turn(tool_calls=(("submit", {"answer": "5"}),)),
        ],
    )
    episode, _ = await run_episode(spec)
    assert episode.metrics["n_sessions"] == 3
    assert episode.calls[1].tool_calls[0].result == "tried A"
    prompt = spec.policies["script"].calls[-1].prompt_text
    assert "[Session 3 of 3] — final session: you must submit before it ends" in prompt
    assert "[Your notes]\ntried A\ntried B" in prompt
    assert "Save progress to notes regularly" in prompt
    assert "end without warning" in prompt


class NoDeltaRenderer(FakeRenderer):
    supports_delta = False


@pytest.mark.parametrize("renderer_type", [FakeRenderer, NoDeltaRenderer])
async def test_tail_carries_only_own_parsed_replies_and_survives_rerender(
    renderer_type: type[FakeRenderer],
) -> None:
    config = MultiSessionConfig(
        sessions=2,
        carry="tail",
        tail_tokens=500,
        tools=("submit", "end_session", "read_scratchpad", "write_scratchpad"),
    )
    turns = [
        Turn(
            "First visible",
            thinking="First thinking",
            tool_calls=(("write_scratchpad", {"content": "TOOL-ONLY-MARKER"}),),
        ),
        Turn("Last visible", thinking="Last thinking", tool_calls=(("end_session", {}),)),
        Turn("Continue", tool_calls=(("read_scratchpad", {"agent_id": "solver0"}),)),
        Turn(tool_calls=(("submit", {"answer": "5"}),)),
    ]
    spec = make_spec(config, turns)
    renderer = renderer_type()
    spec.policies["script"] = ScriptedPolicy(
        "script", renderer, turns_by_agent(renderer, {"solver0": turns})
    )
    spec.renderers["script"] = renderer_type
    episode, buffers = await run_episode(spec)
    assert episode.ok and episode.metrics["n_sessions"] == 2
    expected = (
        "[Tail of your previous session]\nFirst thinking\n\nFirst visible"
        "\n\nLast thinking\n\nLast visible"
    )
    prompts = spec.policies["script"].calls
    for ctx in prompts[2:]:
        first_user = ctx.prompt_text.split("⟨user⟩", 1)[1].split("⟨eot⟩", 1)[0]
        assert expected in first_user
        assert "TOOL-ONLY-MARKER" not in first_user
        assert "⟨call⟩" not in first_user and "⟨think⟩" not in first_user
        assert ctx.prompt_ids[-1] == S["asst"]
    for call, ctx in zip(episode.calls, prompts, strict=True):
        assert tuple(buffers[call.segment_id][: call.prompt_len]) == ctx.prompt_ids
        assert (
            tuple(
                buffers[call.segment_id][
                    call.prompt_len : call.prompt_len + len(call.completion_ids)
                ]
            )
            == call.completion_ids
        )


async def test_both_mid_session_compaction_then_carry_preserves_notes_and_labels() -> None:
    config = MultiSessionConfig(sessions=2, carry="both", compact_threshold=2000)
    spec = make_spec(config, [])
    renderer = FakeRenderer()
    actions = 0

    def script(ctx: ScriptCtx) -> Sequence[int]:
        nonlocal actions
        if ctx.meta.purpose in ("compact", "carry"):
            assert "plus your saved notes" in ctx.prompt_text
            return renderer.encode_completion(f"{ctx.meta.purpose} summary")
        actions += 1
        if actions == 1:
            return renderer.encode_completion(
                "w" * 1800, tool_calls=(("write_notes", {"content": "NOTE1"}),)
            )
        if actions == 2:
            return renderer.encode_completion(tool_calls=(("end_session", {}),))
        return renderer.encode_completion(tool_calls=(("submit", {"answer": "5"}),))

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, _ = await run_episode(spec)
    assert [call.purpose for call in episode.calls] == [
        Purpose.ACT,
        Purpose.COMPACT,
        Purpose.ACT,
        Purpose.CARRY,
        Purpose.ACT,
    ]
    prompts = spec.policies["script"].calls
    assert "[Summary of your earlier work]\ncompact summary" in prompts[2].prompt_text
    assert "[Session 1 of 2]" in prompts[2].prompt_text
    assert "[Previous session summary]\ncarry summary" in prompts[4].prompt_text
    assert (
        "[Your notes]\nNOTE1" in prompts[2].prompt_text
        and "[Your notes]\nNOTE1" in prompts[4].prompt_text
    )


@pytest.mark.parametrize("template", ["{unknown}", "{}", "{agent_id.missing}", "{"])
def test_invalid_session_templates_fail_at_config_time(template: str) -> None:
    with pytest.raises(ConfigError, match="system_prompt"):
        MultiSessionConfig(system_prompt=template)


@pytest.mark.parametrize(
    "boxed,expected", [(r"\boxed{{}}", r"\boxed{}"), (r"\boxed{{{{}}}}", r"\boxed{{}}")]
)
async def test_session_prompt_formats_once_including_literal_braces(
    boxed: str, expected: str
) -> None:
    config = MultiSessionConfig(
        carry="notes",
        session_tokens=3000,
        system_prompt=boxed + "; literal {{sessions}}; {sessions:02d}; {session_tokens:05d}; "
        "{agent_id}; {carry_instructions}",
    )
    spec = make_spec(config, [Turn(tool_calls=(("submit", {"answer": "5"}),))])
    episode, _ = await run_episode(spec)
    assert episode.ok
    assert (
        expected + "; literal {sessions}; 03; 03000; solver0;"
        in spec.policies["script"].calls[0].prompt_text
    )


def test_tail_defaults_end_on_budget_and_self_refine_forbids_early_submit() -> None:
    role = MultiSessionProtocol(MultiSessionConfig(carry="tail")).roles()[0]
    assert "end_session" not in role.tools and "prefill" not in role.system_prompt
    assert "Sessions end when their token budget runs out" in role.system_prompt
    config = resolve_protocol("self_refine_s2")[1]
    assert "Do not call submit in session 1" in config.system_prompt


async def test_last_end_session_uses_last_session_final_instruction() -> None:
    config = MultiSessionConfig(sessions=2, carry="notes")
    spec = make_spec(config, [])
    renderer = FakeRenderer()

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose == "final":
            assert "This is your last session. Submit your final answer now." in ctx.prompt_text
            assert "You have run out of budget" not in ctx.prompt_text
            return [*renderer.encode_text('{"answer": "5"}}'), S["/call"], S["eot"]]
        return renderer.encode_completion(tool_calls=(("end_session", {}),))

    spec.policies["script"] = ScriptedPolicy("script", renderer, script)
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer == "5"
