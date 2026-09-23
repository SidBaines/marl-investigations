"""Saved model output stays inert: every data value is escaped into static HTML.

Native links and details elements work offline without a script, asset server,
or tokenizer. Thinking is separated only from delimiters retained in Call.text;
the records do not store parsed thinking independently of the completion.
"""

from __future__ import annotations

import json
import re
from array import array
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from typing import Any

from marli.interact.types import AgentInfo, Call, Episode, Purpose, record_to_dict

_CSS = """
:root { color-scheme: light dark; font: 15px/1.5 system-ui, sans-serif; }
body { margin: 2rem auto; padding: 0 1rem; max-width: 1500px; }
h1, h2, h3, h4 { line-height: 1.2; }
a { color: light-dark(#1355a2, #90c4ff); }
table { border-collapse: collapse; width: 100%; margin: .7rem 0; }
th, td { border: 1px solid #8886; padding: .5rem; text-align: left; vertical-align: top; }
th { background: #8882; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: .5rem 0; }
td, li, summary { overflow-wrap: anywhere; }
details { border: 1px solid #8886; border-radius: .3rem; padding: .5rem; margin: .4rem 0; }
summary { cursor: pointer; }
.scroll { overflow-x: auto; }
.tick-grid { table-layout: fixed; min-width: 650px; }
.tick-grid th:first-child { width: 4rem; }
.episode { border-top: 3px solid #8886; margin-top: 3rem; padding-top: 1rem; }
.episode:target { border-top-color: #458de1; }
.badge { display: inline-block; background: #8882; border-radius: .3rem;
         padding: 0 .35rem; margin: .1rem; font-size: .85em; }
.forced { background: #d98e0033; }
.error, .diff-del { background: #d7383830; }
.diff-add { background: #1baf5930; }
.diff span { display: block; min-height: 1.5em; }
.muted { opacity: .75; }
.truncated { border-left: 3px solid #d98e00; padding-left: .6rem; }
.metadata { max-width: 70rem; }
"""


def _text(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _e(value: Any) -> str:
    return escape(_text(value), quote=True)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    head = "".join(f'<th scope="col">{_e(label)}</th>' for label in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(v)}</td>" for v in row) + "</tr>" for row in rows)
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _mapping(values: Mapping[str, Any]) -> str:
    return _table(("Field", "Value"), values.items())


def _clip_parts(parts: list[tuple[str, str]], limit: int) -> list[tuple[str, str]]:
    """Keep a single head/tail character budget across the completion's channels."""
    size = sum(len(text) for _, text in parts)
    if size <= limit:
        return parts
    head, tail = (limit + 1) // 2, size - limit // 2
    result: list[tuple[str, str]] = []
    offset = 0
    for kind, text in parts:
        end = offset + len(text)
        if offset < head:
            result.append((kind, text[: head - offset]))
        if offset <= head < end:
            result.append(("truncated", f"[… truncated {size - limit} characters …]"))
        if end > tail:
            result.append((kind, text[max(0, tail - offset) :]))
        offset = end
    return result


def _pre(text: str, limit: int) -> str:
    return "".join(
        f'<pre class="{kind}">{_e(part)}</pre>'
        for kind, part in _clip_parts([("text", text)], limit)
    )


def _completion_parts(call: Call, renderer: str | None) -> list[tuple[str, str]]:
    text = call.text
    if not call.segment_id:
        return [("visible", text)]  # API reasoning is absent from the persisted Call.
    if renderer and renderer.startswith("gpt_oss_"):
        parts = []
        for frame in text.split("<|start|>assistant"):
            header, separator, body = frame.partition("<|message|>")
            if not separator:
                if frame:
                    parts.append(("visible", frame))
                continue
            body = re.sub(r"<\|(?:end|return|call)\|>$", "", body)
            if "to=functions." not in header:
                channel = "thinking" if "<|channel|>analysis" in header else "visible"
                parts.append((channel, body))
        return parts
    if renderer == "fake":
        opener, closer = "⟨think⟩", "⟨/think⟩"
        text = text.removesuffix("⟨eot⟩")
        tool_pattern = r"⟨call⟩.*?(?:⟨/call⟩|$)"
    elif renderer and renderer.startswith("qwen3"):
        opener, closer = "<think>", "</think>"
        text = text.removesuffix("<|im_end|>").removesuffix("<|endoftext|>")
        tool_pattern = r"<tool_call>.*?(?:</tool_call>|$)"
        if renderer == "qwen3_5" and call.purpose == Purpose.ACT and not call.forced:
            # This renderer prefills the opener, which is not in Call.text.
            text = opener + text
    else:
        return [("visible", text)]
    parts = []
    while opener in text:
        before, _, text = text.partition(opener)
        parts.append(("visible", before))
        thinking, _, text = text.partition(closer)
        parts.append(("thinking", thinking))
    parts.append(("visible", text))
    return [
        (kind, re.sub(tool_pattern, "", part, flags=re.DOTALL) if kind == "visible" else part)
        for kind, part in parts
    ]


def _call(call: Call, renderer: str | None, limit: int, include_thinking: bool) -> str:
    forced = '<span class="badge forced">forced</span>' if call.forced else ""
    gen_tokens = len(call.completion_ids) if call.segment_id else call.usage.completion_tokens
    parts = [
        '<details class="call"><summary>',
        f'{_e(call.call_id)} <span class="badge">{_e(call.purpose)}</span>{forced} ',
        f"{_e(call.termination)} · gen tokens: {_e(gen_tokens)}",
        "</summary>",
        _mapping(
            {
                "agent": call.agent_id,
                "role": call.role,
                "policy": call.policy_id,
                "policy_version": call.policy_version,
                "seq": call.seq,
                "tick": call.tick,
                "session": call.session_idx,
                "segment": call.segment_id,
            }
        ),
    ]
    channels = [
        (kind, text)
        for kind, text in _completion_parts(call, renderer)
        if text and (include_thinking or kind != "thinking")
    ]
    for kind, text in _clip_parts(channels, limit):
        if kind == "thinking":
            parts.append(
                '<details class="thinking"><summary>Thinking</summary>'
                f"<pre>{_e(text)}</pre></details>"
            )
        elif kind == "truncated":
            parts.append(f'<p class="truncated">{_e(text)}</p>')
        else:
            parts.append(f'<h4>Visible text</h4><pre class="visible">{_e(text)}</pre>')
    if call.tool_calls:
        parts.append("<h4>Tool calls</h4>")
    for tool in call.tool_calls:
        css = "tool error" if tool.error is not None or not tool.parsed_ok else "tool"
        parts.append(f'<section class="{css}"><h4>{_e(tool.index)}: {_e(tool.name)}</h4>')
        parts.append("<strong>Arguments</strong>" + _pre(_text(tool.arguments), limit))
        if not tool.parsed_ok:
            parts.append("<strong>Unparsed call</strong>" + _pre(tool.raw, limit))
        parts.append("<strong>Result</strong>" + _pre(tool.result, limit))
        if tool.error is not None:
            parts.append("<strong>Error</strong>" + _pre(tool.error, limit))
        parts.append("</section>")
    parts.append("<h4>Workspace reads</h4>")
    parts.append(
        _table(
            ("Writer", "Key", "Version", "Via"),
            ((r.writer, r.key, r.version, r.via) for r in call.reads),
        )
    )
    parts.append("<h4>Usage</h4>" + _mapping(record_to_dict(call.usage)))
    parts.append("</details>")
    return "".join(parts)


def _agent_tree(episode: Episode, agents: Sequence[AgentInfo]) -> str:
    children: dict[str | None, list[AgentInfo]] = defaultdict(list)
    for agent in agents:
        children[agent.parent].append(agent)

    def branch(parent: str | None) -> str:
        result = ["<ul>"]
        for agent in children[parent]:
            result.append(f"<li><strong>{_e(agent.agent_id)}</strong> ({_e(agent.role)})")
            segments = [s for s in episode.segments if s.agent_id == agent.agent_id]
            sessions = sorted(
                {s.session_idx for s in segments}
                | {c.session_idx for c in episode.calls if c.agent_id == agent.agent_id}
            )
            result.append("<ul>")
            for session in sessions:
                result.append(f"<li>Session {_e(session)}<ul>")
                for segment in segments:
                    if segment.session_idx == session:
                        result.append(
                            f"<li>{_e(segment.segment_id)} · start_reason: "
                            f"{_e(segment.start_reason)} · "
                            f"carry_from: {_e(segment.carry_from)}</li>"
                        )
                result.append("</ul></li>")
            result.append("</ul>")
            if children[agent.agent_id]:
                result.append(branch(agent.agent_id))
            result.append("</li>")
        result.append("</ul>")
        return "".join(result)

    return '<div class="agent-tree">' + branch(None) + "</div>"


def _timeline(
    episode: Episode, agents: Sequence[AgentInfo], limit: int, include_thinking: bool
) -> str:
    renderers = {s.segment_id: s.renderer for s in episode.segments}
    calls = sorted(episode.calls, key=lambda c: c.seq)
    details = {
        call.call_id: _call(call, renderers.get(call.segment_id), limit, include_thinking)
        for call in calls
    }
    if not any(call.tick is not None for call in calls):
        return '<div class="async-timeline">' + "".join(details.values()) + "</div>"
    cells: dict[tuple[int | None, str], list[Call]] = defaultdict(list)
    for call in calls:
        cells[call.tick, call.agent_id].append(call)
    ticks = sorted({c.tick for c in calls if c.tick is not None})
    parts = [
        '<div class="scroll"><table class="tick-grid"><caption>Lockstep ticks × agents</caption>'
    ]
    parts.append('<thead><tr><th scope="col">Tick</th>')
    parts.extend(f'<th scope="col">{_e(a.agent_id)}</th>' for a in agents)
    parts.append("</tr></thead><tbody>")
    for tick in ticks:
        parts.append(f'<tr><th scope="row">{_e(tick)}</th>')
        for agent in agents:
            parts.append(
                "<td>" + "".join(details[c.call_id] for c in cells[tick, agent.agent_id]) + "</td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def line_diff(previous: str, current: str) -> list[tuple[str, str]]:
    """A shortest line edit script from the longest common subsequence (LCS)."""
    old, new = previous.splitlines(), current.splitlines()
    lengths = [array("I", [0]) * (len(new) + 1) for _ in range(len(old) + 1)]
    for i in range(len(old) - 1, -1, -1):
        for j in range(len(new) - 1, -1, -1):
            lengths[i][j] = (
                1 + lengths[i + 1][j + 1]
                if old[i] == new[j]
                else max(lengths[i + 1][j], lengths[i][j + 1])
            )
    result = []
    i = j = 0
    while i < len(old) and j < len(new):
        if old[i] == new[j]:
            result.append((" ", old[i]))
            i, j = i + 1, j + 1
        elif lengths[i + 1][j] >= lengths[i][j + 1]:
            result.append(("-", old[i]))
            i += 1
        else:
            result.append(("+", new[j]))
            j += 1
    result.extend(("-", line) for line in old[i:])
    result.extend(("+", line) for line in new[j:])
    return result


def _workspace(episode: Episode) -> str:
    previous: dict[tuple[str, str], tuple[int, str]] = {}
    parts = ['<div class="workspace-log">']
    classes = {" ": "diff-same", "+": "diff-add", "-": "diff-del"}
    for write in sorted(episode.workspace_log, key=lambda w: w.seq):
        version, content = previous.get((write.writer, write.key), (0, ""))
        parts.append(
            f'<details class="write"><summary>{_e(write.writer)} · {_e(write.key)} · '
            f"v{_e(version)} → v{_e(write.version)} · tick {_e(write.tick)} · seq {_e(write.seq)}"
            '</summary><pre class="diff">'
        )
        parts.extend(
            f'<span class="{classes[operation]}">{operation} {_e(line)}</span>'
            for operation, line in line_diff(content, write.content)
        )
        parts.append("</pre></details>")
        previous[write.writer, write.key] = (write.version, write.content)
    parts.append("</div>")
    return "".join(parts)


def render_html(
    episodes: Sequence[Episode], *, max_chars_per_call: int = 20000, include_thinking: bool = True
) -> str:
    """Render each call once, using numeric anchors independent of untrusted IDs."""
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">",
        f"<title>marli episode viewer</title><style>{_CSS}</style></head><body>",
        '<h1 id="index">Episode viewer</h1>',
        f"<p>{len(episodes)} episodes. Expand calls and writes to inspect details.</p>",
        '<div class="scroll"><table class="episode-index"><thead><tr>',
    ]
    headers = (
        "episode_id",
        "task_id",
        "protocol",
        "final answer",
        "_system correct",
        "n_agents",
        "total_gen",
        "cp_tokens",
        "calls",
        "peak_ctx",
        "ok/errors",
    )
    parts.extend(f'<th scope="col">{_e(h)}</th>' for h in headers)
    parts.append("</tr></thead><tbody>")
    for index, episode in enumerate(episodes):
        parts.append(f'<tr><td><a href="#episode-{index}">{_e(episode.episode_id)}</a></td>')
        values = (
            episode.task_id,
            episode.protocol,
            episode.outcome.final_answer,
            episode.grades.get("_system", {}).get("correct"),
            *(
                episode.metrics.get(k)
                for k in ("n_agents", "total_gen", "cp_tokens", "calls", "peak_ctx")
            ),
            f"ok={episode.ok}; " + "; ".join(episode.errors),
        )
        parts.extend(f"<td>{_e(v)}</td>" for v in values)
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    for index, episode in enumerate(episodes):
        agents = sorted(episode.agents, key=lambda a: a.seat_key)
        parts.extend(
            [
                f'<article class="episode" id="episode-{index}">',
                '<a href="#index">Back to index</a>',
                f"<h2>{_e(episode.episode_id)}</h2>",
                '<div class="metadata"><h3>Outcome</h3>',
                _mapping(record_to_dict(episode.outcome)),
                "<h3>Grades per agent</h3>" + _mapping(episode.grades),
                "<h3>limits_hit</h3>" + _mapping(episode.limits_hit),
                "<h3>Metrics</h3>" + _mapping(episode.metrics),
                f'<p class="{"error" if not episode.ok or episode.errors else "status"}">'
                f"ok={_e(episode.ok)}; errors: {_e(episode.errors)}</p></div>",
                "<h3>Agent tree</h3>" + _agent_tree(episode, agents),
                "<h3>Timeline</h3>",
            ]
        )
        if include_thinking and any(not c.segment_id for c in episode.calls):
            parts.append('<p class="muted">API thinking is not present in saved call records.</p>')
        parts.append(_timeline(episode, agents, max_chars_per_call, include_thinking))
        parts.append("<h3>Workspace log</h3>" + _workspace(episode) + "</article>")
    parts.append("</body></html>\n")
    return "".join(parts)
