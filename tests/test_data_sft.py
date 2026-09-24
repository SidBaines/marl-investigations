"""SFT rejection and masks are checked against actual scripted runtime traces."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _marli_test_envs import ArithEnv

from marli.cli.main import main
from marli.data import sft
from marli.data.sft import DataSFTConfig, SFTSet, read_datums
from marli.errors import ConfigError
from marli.eval.rollout import EpisodeSet
from marli.interact.configs import build_protocol
from marli.interact.limits import Limits
from marli.interact.records import write_episode
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.types import Episode, Purpose
from marli.model import ModelSpec, load_model
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn, from_callable
from marli.render.fake import FakeRenderer
from marli.rundir import RunDir, RunStatus
from marli.verbs import run_verb


def teacher_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()

    def turn(ctx: ScriptCtx) -> Turn:
        return Turn(tool_calls=(("submit", {"answer": "5"}),))

    return ScriptedPolicy("teacher", renderer, from_callable(turn, renderer))


async def teacher_episode(
    index: int = 0,
    *,
    first: Turn | None = None,
    sessions: bool = False,
    on_no_tool_call: str = "nudge",
) -> tuple[Episode, dict[str, list[int]]]:
    renderer = FakeRenderer()

    def turn(ctx: ScriptCtx) -> Turn:
        if ctx.meta.call_index == 0 and first is not None:
            return first
        return Turn(thinking="x" * index, tool_calls=(("submit", {"answer": "5"}),))

    policy = ScriptedPolicy("teacher", renderer, from_callable(turn, renderer))
    env = ArithEnv()
    protocol = build_protocol(
        "multi_session" if sessions else "single",
        {"sessions": 2, "carry": "notes"} if sessions else {"env_tools": ["write_scratchpad"]},
    )
    return await run_episode(
        EpisodeSpec(
            protocol,
            env,
            env.task,
            {"solver": "teacher"},
            {"teacher": policy},
            {"teacher": FakeRenderer},
            protocol.adjust_limits(Limits(on_no_tool_call=on_no_tool_call)),
            group_id=f"teacher{index}",
            episode_idx=index,
        )
    )


def save_teacher(
    root: Path,
    episodes: Sequence[tuple[Episode, dict[str, list[int]]]],
    *,
    record_tokens: bool = True,
    on_no_tool_call: str = "nudge",
) -> EpisodeSet:
    with RunDir(
        root, kind="eval rollout", manifest_name="episodes.json", config_hash="teacher"
    ) as r:
        for episode, buffers in episodes:
            write_episode(r.append_row, episode, buffers, record_tokens=record_tokens)
        source = EpisodeSet(
            root=root,
            episodes="episodes.jsonl",
            tokens="tokens.jsonl" if record_tokens else None,
            n=len(episodes),
            n_failed=sum(not e.ok for e, _ in episodes),
            protocol="single",
            protocol_config={},
            env="arith",
            env_config={},
            taskset="unused",
            policies={"teacher": "scripted:test_data_sft:teacher_policy"},
            usage={},
            cost_usd=0,
            meta={"limits": {"on_no_tool_call": on_no_tool_call}},
        )
        return r.finalize(source)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], ModelSpec]:
    def load(name: str) -> ModelSpec:
        return replace(load_model(name), renderer="fake")

    monkeypatch.setattr(sft, "load_model", load)
    return load


async def build_set(root: Path, source: EpisodeSet, **kwargs: Any) -> SFTSet:
    result = await run_verb(
        "data sft",
        DataSFTConfig(episodes=str(source.root), student_model="qwen3_8b", **kwargs),
        out=root,
    )
    assert isinstance(result.handle, SFTSet)
    return result.handle


async def test_episode_filters_counts_and_packed_roundtrip(
    tmp_path: Path,
    fake_model: Callable[[str], ModelSpec],
) -> None:
    good = await teacher_episode()
    wrong, b1 = await teacher_episode(1)
    failed, b2 = await teacher_episode(2)
    wrong = replace(wrong, grades={"_system": {"correct": 0.0}})
    failed = replace(failed, ok=False)
    source = save_teacher(tmp_path / "teacher", [good, (wrong, b1), (failed, b2)])
    data = await build_set(tmp_path / "sft", source)
    assert data.counts["episodes"] == {
        "input": 3,
        "kept": 1,
        "dropped_not_ok": 1,
        "dropped_incorrect": 1,
    }
    assert data.n == 1 and data.roles == ("solver",)
    assert data.teacher_policies == source.policies
    assert data.tokenizer_sha == FakeRenderer.tokenizer_sha
    assert data.inputs[0].sha256 == source.sha256()
    [datum] = read_datums(SFTSet.load(data.root))
    assert datum.tokens == tuple(good[1][datum.segment_id])
    assert set(datum.advantages) == set(datum.logprobs) == {0.0}
    assert data.n_tokens == len(datum.tokens)
    assert data.n_action_tokens == len(good[0].calls[0].completion_ids)
    row = json.loads(data.file("datums").read_text())
    assert isinstance(row["tokens"], str) and isinstance(row["mask"], str)
    all_data = await build_set(
        tmp_path / "all",
        source,
        require_correct=False,
        require_ok=False,
    )
    assert all_data.n == 3


@pytest.mark.parametrize("sessions", [False, True])
async def test_exact_completion_masks_and_segment_boundaries(
    tmp_path: Path,
    fake_model: Callable[[str], ModelSpec],
    sessions: bool,
) -> None:
    first = (
        Turn(tool_calls=(("end_session", {}),))
        if sessions
        else Turn(
            tool_calls=(("write_scratchpad", {"content": "a note"}),),
            stop=False,
        )
    )
    episode, buffers = await teacher_episode(first=first, sessions=sessions)
    assert episode.ok and episode.grades["_system"]["correct"] == 1
    source = save_teacher(tmp_path / "teacher", [(episode, buffers)])
    data = await build_set(tmp_path / "sft", source)
    datums = list(read_datums(data))
    assert len(datums) == (2 if sessions else 1)
    for datum in datums:
        calls = [c for c in episode.calls if c.segment_id == datum.segment_id and c.completion_ids]
        end = max(c.prompt_len + len(c.completion_ids) for c in calls)
        assert datum.tokens == tuple(buffers[datum.segment_id][:end])
        expected = [0.0] * (end - 1)
        for call in calls:
            start = call.prompt_len - 1
            expected[start : start + len(call.completion_ids)] = [1.0] * len(call.completion_ids)
        assert datum.mask == tuple(expected)
    assert data.counts["datums"]["kept"] == len(datums)
    short = await build_set(tmp_path / "short", source, max_len=2)
    assert short.n == 0 and list(read_datums(short)) == []
    assert short.counts["datums"]["dropped_max_len"] == len(datums)


@pytest.mark.parametrize("reason", ["unparsed", "nudge", "forced"])
async def test_agent_conformance_rejection(
    tmp_path: Path,
    fake_model: Callable[[str], ModelSpec],
    reason: str,
) -> None:
    first = (
        Turn(raw_tool_bodies=("bad-json",)) if reason == "unparsed" else Turn(content="thinking")
    )
    episode, buffers = await teacher_episode(first=first)
    if reason == "forced":
        episode = replace(
            episode, calls=(replace(episode.calls[0], forced=True), *episode.calls[1:])
        )
    source = save_teacher(tmp_path / "teacher", [(episode, buffers)])
    data = await build_set(tmp_path / "sft", source)
    assert data.n == 0
    assert data.counts["agents"][f"dropped_{reason}"] == 1
    assert (await build_set(tmp_path / "all", source, require_conformant=False)).n == 1


async def test_nudge_filter_respects_text_answers_and_non_action_purposes(
    tmp_path: Path,
    fake_model: Callable[[str], ModelSpec],
) -> None:
    episode = await teacher_episode(first=Turn(content="5"), on_no_tool_call="final_text_as_answer")
    source = save_teacher(tmp_path / "teacher", [episode], on_no_tool_call="final_text_as_answer")
    assert (await build_set(tmp_path / "sft", source)).n == 1
    ep, buffers = await teacher_episode(first=Turn(content="summary"))
    ep = replace(ep, calls=(replace(ep.calls[0], purpose=Purpose.COMPACT), *ep.calls[1:]))
    source = save_teacher(tmp_path / "compact", [(ep, buffers)])
    assert (await build_set(tmp_path / "compact-sft", source)).n == 1


async def test_role_filter_and_no_trainable_calls(
    tmp_path: Path,
    fake_model: Callable[[str], ModelSpec],
) -> None:
    episode, buffers = await teacher_episode()
    source = save_teacher(tmp_path / "teacher", [(episode, buffers)])
    data = await build_set(tmp_path / "sft", source, roles=("worker",))
    assert data.n == 0 and data.counts["agents"]["dropped_role"] == 1
    assert (await build_set(tmp_path / "solver", source, roles=("solver",))).n == 1
    episode = replace(episode, calls=tuple(replace(c, logprobs=None) for c in episode.calls))
    source = save_teacher(tmp_path / "no-logprobs", [(episode, buffers)])
    data = await build_set(tmp_path / "no-trainable", source)
    assert data.n == 0 and data.counts["agents"]["dropped_no_trainable_calls"] == 1


async def test_rejection_is_per_agent_and_roles_do_not_mix(
    tmp_path: Path,
    fake_model: Callable[[str], ModelSpec],
) -> None:
    good, good_buffers = await teacher_episode()
    bad, bad_buffers = await teacher_episode(1, first=Turn(raw_tool_bodies=("bad-json",)))
    segment_ids = {s.segment_id: s.segment_id.replace("solver0", "worker0") for s in bad.segments}
    mixed = replace(
        good,
        agents=good.agents
        + tuple(replace(a, agent_id="worker0", role="worker") for a in bad.agents),
        segments=good.segments
        + tuple(
            replace(s, agent_id="worker0", segment_id=segment_ids[s.segment_id])
            for s in bad.segments
        ),
        calls=good.calls
        + tuple(
            replace(
                c,
                episode_id=good.episode_id,
                agent_id="worker0",
                role="worker",
                segment_id=segment_ids[c.segment_id],
                seq=c.seq + 100,
            )
            for c in bad.calls
        ),
    )
    source = save_teacher(
        tmp_path / "teacher",
        [
            (
                mixed,
                {
                    **good_buffers,
                    **{segment_ids[key]: value for key, value in bad_buffers.items()},
                },
            )
        ],
    )
    data = await build_set(tmp_path / "conformant", source)
    [kept] = read_datums(data)
    assert kept.agent_id == "solver0" and kept.tokens == tuple(good_buffers[kept.segment_id])
    assert data.counts["agents"]["kept"] == data.counts["agents"]["dropped_unparsed"] == 1
    worker = await build_set(
        tmp_path / "worker", source, require_conformant=False, roles=("worker",)
    )
    [kept_worker] = read_datums(worker)
    assert kept_worker.agent_id == "worker0" and worker.roles == ("worker",)
    assert worker.counts["agents"]["dropped_role"] == 1


async def test_tokenizer_and_missing_token_guards(
    tmp_path: Path,
    fake_model: Callable[[str], ModelSpec],
) -> None:
    episode, buffers = await teacher_episode()
    episode = replace(
        episode, segments=tuple(replace(s, tokenizer_sha="wrong") for s in episode.segments)
    )
    source = save_teacher(tmp_path / "teacher", [(episode, buffers)])
    with pytest.raises(ConfigError, match="tokenizer_sha"):
        await build_set(tmp_path / "sft", source)
    source = save_teacher(tmp_path / "no-tokens", [(episode, buffers)], record_tokens=False)
    with pytest.raises(ConfigError, match="record_tokens"):
        await build_set(tmp_path / "missing", source)


async def test_resume_skips_completed_rows_and_recovers_torn_tail(
    tmp_path: Path,
    fake_model: Callable[[str], ModelSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = save_teacher(tmp_path / "teacher", [await teacher_episode(i) for i in range(3)])
    append = RunDir.append_row

    def crash(run: RunDir, name: str, row: dict[str, Any]) -> None:
        append(run, name, row)
        if name == "datums.jsonl":
            raise RuntimeError("crash")

    with monkeypatch.context() as patch:
        patch.setattr(RunDir, "append_row", crash)
        with pytest.raises(RuntimeError, match="crash"):
            await build_set(tmp_path / "sft", source)
    path = tmp_path / "sft/datums.jsonl"
    original = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"tokens":')
    data = await build_set(tmp_path / "sft", source)
    assert path.read_bytes().startswith(original)
    assert data.n == len(list(read_datums(data))) == 3
    result = await run_verb(
        "data sft",
        DataSFTConfig(episodes=str(source.root), student_model="qwen3_8b"),
        out=data.root,
    )
    assert result.status is RunStatus.COMPLETE


def test_data_cli_one_line(
    tmp_path: Path,
    fake_model: Callable[[str], ModelSpec],
    capsys: pytest.CaptureFixture[str],
) -> None:
    import asyncio

    source = save_teacher(tmp_path / "teacher", [asyncio.run(teacher_episode())])
    args = [
        "data",
        "sft",
        f"episodes={source.root}",
        "student_model=qwen3_8b",
        "--out",
        str(tmp_path / "sft"),
    ]
    for status in ("fresh", "complete"):
        assert main(args) == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1
        result = json.loads(lines[0])
        assert result["ok"] and result["kind"] == "sftset" and result["status"] == status
