"""Real runtime traces stay readable, bounded, and inert after offline rendering."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from xml.etree.ElementTree import Element, TreeBuilder

import pytest
from test_interact_agent import RuntimeProtocol, spec_for, submit

from marli.cli.main import main
from marli.errors import ConfigError
from marli.eval.rollout import EpisodeSet
from marli.interact import records
from marli.interact.limits import AgentLimits, CallLimits, Limits
from marli.interact.run import run_episode
from marli.interact.system import ContextSpec
from marli.interact.types import Episode, Purpose, SegmentStart, Write
from marli.policy.scripted import ScriptCtx, ScriptedPolicy, Turn
from marli.render.fake import SPECIAL_IDS, FakeRenderer
from marli.rundir import RunDir, RunStatus
from marli.verbs import run_verb
from marli.viewer.html import line_diff, render_html
from marli.viewer.verb import View, ViewConfig

PAYLOAD = '<script>alert("unsafe & dangerous")</script>'
LONG_TEXT = "HEAD-" + "x" * 1000 + "-TAIL"


class Page(HTMLParser):
    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.builder = TreeBuilder()
        self.feed(html)
        self.close()
        self.root: Element = self.builder.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.builder.start(tag, dict(attrs))
        if tag == "meta":
            self.builder.end(tag)

    def handle_endtag(self, tag: str) -> None:
        self.builder.end(tag)

    def handle_data(self, data: str) -> None:
        self.builder.data(data)


def content(element: Element) -> str:
    return "".join(element.itertext())


@dataclass(frozen=True)
class Saved:
    source: EpisodeSet
    episodes: tuple[Episode, ...]


@pytest.fixture
async def saved(tmp_path: Path) -> Saved:
    peers = RuntimeProtocol(tools=("submit", "write_scratchpad", "read_scratchpad"), count=2)
    peer_spec = spec_for(
        {
            f"solver{i}": [
                Turn(
                    PAYLOAD if i == 0 else LONG_TEXT,
                    thinking=f"private reasoning {i}",
                    tool_calls=(("write_scratchpad", {"content": "keep\nold line"}),),
                ),
                Turn(
                    tool_calls=(
                        ("read_scratchpad", {"agent_id": f"solver{1 - i}"}),
                        ("write_scratchpad", {"content": "keep\nnew line", "mode": "overwrite"}),
                    )
                ),
                Turn(
                    tool_calls=(
                        ("submit", {"answer": "5"}),
                        ("write_scratchpad", {"content": "late"}),
                    )
                ),
            ]
            for i in range(2)
        },
        peers,
    )
    peer_spec.group_id = "viewer-peers"
    peer_trace = await run_episode(peer_spec)
    compact = spec_for(
        {
            "solver0": [
                Turn(tool_calls=(("write_scratchpad", {"content": "x" * 600}),)),
                Turn("A compact summary."),
                submit(),
            ]
        },
        RuntimeProtocol(
            tools=("submit", "write_scratchpad"),
            context=ContextSpec(kind="compaction", compact_threshold=1000, compact_reserve=512),
        ),
    )
    compact.group_id = "viewer-compaction"
    compact.limits.session.carry_max_tokens = 256
    compact_trace = await run_episode(compact)
    renderer = FakeRenderer()
    forced = spec_for(
        {"solver0": []},
        limits=Limits(agent=AgentLimits(max_calls=1), call=CallLimits(min_call_tokens=1)),
    )
    forced.group_id = "viewer-forced"

    def script(ctx: ScriptCtx) -> Sequence[int]:
        if ctx.meta.purpose == "act":
            return renderer.encode_completion("Still working.")
        return [*renderer.encode_text('{"answer": "5"}}'), SPECIAL_IDS["/call"], SPECIAL_IDS["eot"]]

    forced.policies["script"] = ScriptedPolicy("script", renderer, script)
    forced_trace = await run_episode(forced)
    assert all(trace[0].ok for trace in (peer_trace, compact_trace, forced_trace))
    assert any(c.reads for c in peer_trace[0].calls)
    assert compact_trace[0].segments[1].start_reason == SegmentStart.COMPACTION
    assert forced_trace[0].calls[-1].purpose == Purpose.FINAL
    source = EpisodeSet(
        root=tmp_path / "episodes",
        episodes="episodes.jsonl",
        tokens="tokens.jsonl",
        n=3,
        n_failed=0,
        protocol="viewer-test",
        protocol_config={},
        env="scratch",
        env_config={},
        taskset="unused.json",
        policies={"script": "scripted"},
        usage={},
        cost_usd=0,
    )
    with RunDir(
        source.root, kind="eval rollout", manifest_name=source.MANIFEST, config_hash="test"
    ) as run:
        for episode, buffers in (peer_trace, compact_trace, forced_trace):
            records.write_episode(run.append_row, episode, buffers, record_tokens=True)
        source = run.finalize(source)
    return Saved(source, (peer_trace[0], compact_trace[0], forced_trace[0]))


def invoke(capsys: pytest.CaptureFixture[str], *args: str, code: int = 0) -> dict[str, object]:
    assert main(list(args)) == code
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    result = json.loads(captured.out)
    assert result["ok"] is (code == 0)
    return result


def test_cli_runtime_traces_and_idempotency(
    saved: Saved, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "view"
    # The shared CLI accepts dataclass overrides, not per-field --flags.
    args = ("view", f"episodes={saved.source.root}", "--out", str(out))
    result = invoke(capsys, *args)
    assert result["kind"] == "view" and result["status"] == "fresh"
    handle = View.load(str(result["manifest"]))
    assert handle.file("html") == out / "index.html"
    assert handle.inputs[0].resolve() == saved.source.manifest_path
    html = handle.file("html").read_text()
    page = Page(html).root
    for episode in saved.episodes:
        assert episode.episode_id in html
        for agent in episode.agents:
            assert agent.agent_id in html
    assert page.find('.//table[@class="tick-grid"]') is not None
    assert "compaction" in html and "carry_from: solver0/g0" in html
    assert '<span class="badge forced">forced</span>' in html
    assert "private reasoning 0" in html
    assert page.find('.//details[@class="thinking"]') is not None
    assert "tool call after control tool" in html
    assert page.find('.//section[@class="tool error"]') is not None
    assert "notify" in html and "pull" in html
    assert "prompt_tokens" in html and "completion_tokens" in html
    diffs = page.findall('.//details[@class="write"]')
    second = next(d for d in diffs if "solver0 · scratchpad · v1 → v2" in content(d))
    assert content(second.find('.//span[@class="diff-del"]')) == "- old line"
    assert content(second.find('.//span[@class="diff-add"]')) == "+ new line"
    assert escape(PAYLOAD, quote=True) in html and PAYLOAD not in html
    assert PAYLOAD in content(page)
    assert page.findall(".//script") == []
    assert page.findall(".//*[@src]") == [] and page.findall(".//link") == []
    assert all(a.attrib["href"].startswith("#") for a in page.findall(".//a"))
    assert "innerHTML" not in html
    before = handle.file("html").stat().st_mtime_ns
    assert invoke(capsys, *args)["status"] == "complete"
    assert handle.file("html").stat().st_mtime_ns == before
    assert handle.file("html").read_text() == html


def test_cli_truncation_and_thinking_disabled(
    saved: Saved, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = invoke(
        capsys,
        "view",
        f"episodes={saved.source.manifest_path}",
        "max_chars_per_call=100",
        "include_thinking=false",
        "--out",
        str(tmp_path / "short"),
    )
    html = View.load(str(result["manifest"])).file("html").read_text()
    assert "truncated" in content(Page(html).root)
    assert "HEAD-" in html and "-TAIL" in html and LONG_TEXT not in html
    assert "private reasoning" not in html and '<details class="thinking">' not in html


@pytest.mark.parametrize("explicit", [False, True])
async def test_selection_stops_before_unselected_rows_and_ignores_tokens(
    saved: Saved, tmp_path: Path, explicit: bool
) -> None:
    chosen = saved.episodes[1] if explicit else saved.episodes[0]
    rows = saved.source.file("episodes").read_text().splitlines()
    count = 2 if explicit else 1
    saved.source.file("episodes").write_text("\n".join(rows[:count]) + "\ninvalid row\n")
    saved.source.file("tokens").write_text("invalid sidecar\n")
    cfg = ViewConfig(
        str(saved.source.root), episode_ids=[chosen.episode_id] if explicit else [], max_episodes=1
    )
    result = await run_verb("view", cfg, out=tmp_path / "selected")
    html = result.handle.file("html").read_text()
    assert chosen.episode_id in html
    assert all(e.episode_id not in html for e in saved.episodes if e != chosen)


async def test_explicit_ids_override_limit_and_missing_id_fails(
    saved: Saved, tmp_path: Path
) -> None:
    cfg = ViewConfig(
        str(saved.source.root), episode_ids=[e.episode_id for e in saved.episodes], max_episodes=1
    )
    result = await run_verb("view", cfg, out=tmp_path / "selected")
    assert all(e.episode_id in result.handle.file("html").read_text() for e in saved.episodes)
    missing = replace(cfg, episode_ids=["missing"])
    with pytest.raises(ConfigError, match="episode_ids not found"):
        await run_verb("view", missing, out=tmp_path / "missing")
    assert not (tmp_path / "missing" / "view.json").exists()


async def test_resume_after_html_write_failure(
    saved: Saved, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marli.viewer import verb

    cfg = ViewConfig(str(saved.source.root))
    original = verb.atomic_write_text

    def interrupted(path: Path, text: str) -> None:
        original(path, text)
        raise RuntimeError("interrupted before finalizing")

    with monkeypatch.context() as patch:
        patch.setattr(verb, "atomic_write_text", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            await run_verb("view", cfg, out=tmp_path / "resumed")
    assert not (tmp_path / "resumed" / "view.json").exists()
    result = await run_verb("view", cfg, out=tmp_path / "resumed")
    assert result.status is RunStatus.RESUME
    assert saved.episodes[0].episode_id in result.handle.file("html").read_text()


def test_tree_sessions_seat_order_async_and_header(saved: Saved) -> None:
    episode = saved.episodes[0]
    parent, worker = episode.agents
    worker = replace(worker, parent=parent.agent_id, role="worker")
    episode = replace(
        episode,
        agents=(worker, parent),
        segments=(
            *episode.segments,
            replace(
                episode.segments[1],
                segment_id="solver1/g1",
                session_idx=1,
                start_reason=SegmentStart.SESSION,
                carry_from="solver1/g0",
            ),
        ),
        outcome=replace(episode.outcome, aggregation="vote", votes={"5": 2}),
        grades={"_system": {"correct": 1.0}, "solver0": {"correct": 1.0}},
    )
    page = Page(render_html([episode])).root
    columns = page.findall('.//table[@class="tick-grid"]/thead/tr/th')
    assert [content(c) for c in columns] == ["Tick", "solver0", "solver1"]
    root_agent = page.find('.//div[@class="agent-tree"]/ul/li')
    assert content(root_agent.find("strong")) == "solver0"
    assert content(root_agent.find("ul/li/strong")) == "solver1"
    assert "Session 1" in content(root_agent) and "carry_from: solver1/g0" in content(root_agent)
    assert "votes" in content(page) and "vote" in content(page)
    calls = tuple(replace(c, tick=None) for c in reversed(episode.calls))
    async_page = Page(render_html([replace(episode, calls=calls)])).root
    shown = async_page.findall('.//div[@class="async-timeline"]/details/summary')
    assert [content(c).split()[0] for c in shown] == [
        c.call_id for c in sorted(calls, key=lambda c: c.seq)
    ]
    assert async_page.find('.//table[@class="tick-grid"]') is None


def test_lcs_diff_and_independent_workspace_histories(saved: Saved) -> None:
    assert line_diff("a\nb\nc", "b\nc\na") == [("-", "a"), (" ", "b"), (" ", "c"), ("+", "a")]
    assert line_diff("", "one\ntwo") == [("+", "one"), ("+", "two")]
    assert line_diff("one", "") == [("-", "one")]
    episode = replace(
        saved.episodes[0],
        workspace_log=(
            Write("solver0", "scratchpad", 1, "old", 0, 0),
            Write("solver1", "scratchpad", 1, "peer", 0, 1),
            Write("solver0", "notes", 1, "note", 0, 2),
            Write("solver0", "scratchpad", 2, "new", 1, 3),
        ),
    )
    writes = Page(render_html([episode])).root.findall('.//details[@class="write"]')
    last = content(writes[-1])
    assert "- old" in last and "+ new" in last and "peer" not in last and "note" not in last


@pytest.mark.parametrize("field", ["max_episodes", "max_chars_per_call"])
@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_limits(field: str, value: int) -> None:
    with pytest.raises(ConfigError, match=field):
        ViewConfig(**{field: value})


def test_missing_input_and_empty_page(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = invoke(capsys, "view", "--out", str(tmp_path / "missing"), code=2)
    assert result["error"] == "ConfigError"
    assert "0 episodes" in render_html([])


def test_twenty_typical_episodes_fit_offline_page(saved: Saved) -> None:
    html = render_html([replace(saved.episodes[0], episode_id=f"page/e{i}") for i in range(20)])
    assert len(html.encode("utf-8")) < 5_000_000


@pytest.mark.parametrize(
    ("renderer", "text", "visible"),
    [
        ("fake", "⟨think⟩hidden⟨/think⟩shown⟨eot⟩", "shown"),
        ("fake", "⟨think⟩hidden", ""),
        ("qwen3", "<think>hidden</think>shown<|im_end|>", "shown"),
        ("qwen3_5", "hidden</think>shown<|im_end|>", "shown"),
        ("qwen3_5", "hidden", ""),
        (
            "gpt_oss_medium",
            "<|channel|>analysis<|message|>hidden<|end|>"
            "<|start|>assistant<|channel|>final<|message|>shown<|return|>",
            "shown",
        ),
    ],
)
def test_saved_thinking_channels(saved: Saved, renderer: str, text: str, visible: str) -> None:
    episode = saved.episodes[0]
    call = replace(episode.calls[0], text=text, tool_calls=())
    segment = replace(
        next(s for s in episode.segments if s.segment_id == call.segment_id), renderer=renderer
    )
    episode = replace(episode, calls=(call,), segments=(segment,), workspace_log=())
    included = Page(render_html([episode])).root
    thinking = included.find('.//details[@class="thinking"]')
    assert thinking is not None and "hidden" in content(thinking)
    excluded = render_html([episode], include_thinking=False)
    assert "hidden" not in excluded
    if visible:
        assert visible in content(Page(excluded).root)


def test_completion_budget_spans_thinking_and_visible_text(saved: Saved) -> None:
    episode = saved.episodes[0]
    call = replace(
        episode.calls[0],
        text="⟨think⟩" + "a" * 120 + "⟨/think⟩" + "b" * 120 + "⟨eot⟩",
        tool_calls=(),
    )
    page = Page(render_html([replace(episode, calls=(call,))], max_chars_per_call=100)).root
    thinking = page.find('.//details[@class="thinking"]/pre')
    visible = page.find('.//pre[@class="visible"]')
    assert content(thinking) == "a" * 50
    assert content(visible) == "b" * 50
    assert "truncated 140 characters" in content(page)


def test_untrusted_values_cannot_create_markup_or_links(saved: Saved) -> None:
    attack = '"><img src="https://invalid.example" onerror="alert(1)">&\''
    episode = saved.episodes[0]
    call = episode.calls[0]
    tool = replace(
        call.tool_calls[0],
        name=attack,
        arguments={attack: attack},
        result=attack,
        error=attack,
        raw=attack,
        parsed_ok=False,
    )
    call = replace(
        call,
        call_id=attack,
        policy_id=attack,
        text=attack,
        reads=(replace(episode.calls[-1].reads[0], writer=attack, key=attack),),
        tool_calls=(tool,),
    )
    episode = replace(
        episode,
        episode_id=attack,
        task_id=attack,
        protocol=attack,
        calls=(call,),
        outcome=replace(episode.outcome, final_answer=attack, votes={attack: 1}),
        grades={attack: {attack: 1.0}},
        limits_hit={attack: (attack,)},
        errors=(attack,),
        ok=False,
        workspace_log=(Write(attack, attack, 1, attack, 0, 0),),
    )
    html = render_html([episode])
    page = Page(html).root
    assert escape(attack, quote=True) in html
    assert page.findall(".//img") == []
    assert page.findall(".//*[@onerror]") == []
    assert all(a.attrib["href"].startswith("#") for a in page.findall(".//a"))
