"""A deterministic character-level renderer for CPU tests.

Token ids 0..255 are UTF-8 bytes; special tokens start at 256. The format is
deliberately simple but has every feature the harness relies on: a system
header with tool specs, role headers, an end-of-turn stop token, thinking
blocks, tool-call blocks with JSON bodies, grouped tool results, and a
generation header. It supports delta rendering exactly (by construction the
delta path and a full re-render agree), so tests can assert the
"datums use exactly the ids the policy saw" invariant without any model.

Wire format (``⟨x⟩`` = special token)::

    ⟨sys⟩ system text ⟨tools⟩ json(list of tool specs) ⟨eot⟩
    ⟨user⟩ text ⟨eot⟩
    ⟨asst⟩ [⟨think⟩ thinking ⟨/think⟩] content [⟨call⟩ {"name":..,"arguments":{..}} ⟨/call⟩]* ⟨eot⟩
    ⟨tool⟩ ⟨res⟩ result-1 ⟨/res⟩ ⟨res⟩ result-2 ⟨/res⟩ ⟨eot⟩      (consecutive tool msgs grouped)
    ⟨asst⟩   <- generation header ends every prompt / continuation
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from marli.render.base import Msg, ParsedToolCall, ParsedTurn, ToolSpec

SPECIALS = (
    "sys",
    "tools",
    "user",
    "asst",
    "tool",
    "eot",
    "think",
    "/think",
    "call",
    "/call",
    "res",
    "/res",
)
SPECIAL_IDS = {name: 256 + i for i, name in enumerate(SPECIALS)}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_IDS.items()}
VOCAB_SIZE = 256 + len(SPECIALS)
S = SPECIAL_IDS


def _bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


class FakeRenderer:
    name = "fake"
    supports_delta = True
    tokenizer_sha = hashlib.sha256(("fake-v1:" + ",".join(SPECIALS)).encode()).hexdigest()[:16]
    stop_token_ids = (S["eot"],)
    delivery_role = "tool"

    # -- rendering -----------------------------------------------------------------
    def initial(
        self, system: str | None, tools: Sequence[ToolSpec], msgs: Sequence[Msg]
    ) -> list[int]:
        ids: list[int] = []
        if system is not None or tools:
            ids.append(S["sys"])
            ids += _bytes(system or "")
            if tools:
                ids.append(S["tools"])
                ids += _bytes(
                    json.dumps(
                        [
                            {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.parameters,
                            }
                            for t in tools
                        ],
                        sort_keys=True,
                    )
                )
            ids.append(S["eot"])
        ids += self._render_msgs(msgs)
        ids.append(S["asst"])
        return ids

    def continuation(self, last_termination: str, new_msgs: Sequence[Msg]) -> list[int]:
        ids: list[int] = []
        if last_termination != "stop":
            ids.append(S["eot"])  # close the turn the model did not close
        ids += self._render_msgs(new_msgs)
        ids.append(S["asst"])
        return ids

    def forced_tool_prefix(self, tool_name: str) -> list[int]:
        return [S["call"], *_bytes('{"name": ' + json.dumps(tool_name) + ', "arguments": ')]

    def _render_msgs(self, msgs: Sequence[Msg]) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(msgs):
            m = msgs[i]
            if m.role == "tool":
                ids.append(S["tool"])
                while i < len(msgs) and msgs[i].role == "tool":
                    ids += [S["res"], *_bytes(msgs[i].content), S["/res"]]
                    i += 1
                ids.append(S["eot"])
                continue
            if m.role == "user":
                ids += [S["user"], *_bytes(m.content), S["eot"]]
            elif m.role == "assistant":  # only used in re-render mode
                ids.append(S["asst"])
                if m.thinking:
                    ids += [S["think"], *_bytes(m.thinking), S["/think"]]
                ids += _bytes(m.content)
                for tc in m.tool_calls:
                    body = json.dumps({"name": tc.name, "arguments": tc.arguments}, sort_keys=True)
                    ids += [S["call"], *_bytes(body), S["/call"]]
                ids.append(S["eot"])
            elif m.role == "system":
                ids += [S["sys"], *_bytes(m.content), S["eot"]]
            else:
                raise ValueError(f"fake renderer: unknown role {m.role!r}")
            i += 1
        return ids

    # -- scripted-policy helper ----------------------------------------------------
    def encode_completion(
        self,
        content: str = "",
        *,
        thinking: str | None = None,
        tool_calls: Sequence[tuple[str, dict]] = (),
        raw_tool_bodies: Sequence[str] = (),
        stop: bool = True,
    ) -> list[int]:
        """Ids a model would sample for this turn (used by ScriptedPolicy in tests)."""
        ids: list[int] = []
        if thinking is not None:
            ids += [S["think"], *_bytes(thinking), S["/think"]]
        ids += _bytes(content)
        for name, args in tool_calls:
            ids += [
                S["call"],
                *_bytes(json.dumps({"name": name, "arguments": args}, sort_keys=True)),
                S["/call"],
            ]
        for body in raw_tool_bodies:  # deliberately malformed bodies for parser tests
            ids += [S["call"], *_bytes(body), S["/call"]]
        if stop:
            ids.append(S["eot"])
        return ids

    # -- parsing ---------------------------------------------------------------------
    def parse(self, completion_ids: Sequence[int], tools: Sequence[ToolSpec] = ()) -> ParsedTurn:
        ids = list(completion_ids)
        termination = "stop"
        if ids and ids[-1] == S["eot"]:
            ids = ids[:-1]
        else:
            termination = "length"
        thinking: str | None = None
        content_parts: list[str] = []
        calls: list[ParsedToolCall] = []
        i = 0
        while i < len(ids):
            t = ids[i]
            if t == S["think"]:
                j = _find(ids, S["/think"], i + 1)
                end = j if j >= 0 else len(ids)
                thinking = _text(ids[i + 1 : end])
                i = end + 1 if j >= 0 else len(ids)
            elif t == S["call"]:
                j = _find(ids, S["/call"], i + 1)
                if j < 0:
                    raw = _text(ids[i + 1 :])
                    calls.append(ParsedToolCall(name=None, arguments=None, raw=raw, ok=False))
                    if termination == "stop":
                        termination = "malformed"
                    break
                raw = _text(ids[i + 1 : j])
                calls.append(_parse_call(raw))
                i = j + 1
            elif t >= 256:
                i += 1  # stray special: ignore
            else:
                j = i
                while j < len(ids) and ids[j] < 256:
                    j += 1
                content_parts.append(_text(ids[i:j]))
                i = j
        return ParsedTurn(
            content="".join(content_parts),
            thinking=thinking,
            tool_calls=tuple(calls),
            termination=termination,
        )

    def decode(self, ids: Sequence[int]) -> str:
        out: list[str] = []
        buf: list[int] = []
        for t in ids:
            if t < 256:
                buf.append(t)
            else:
                if buf:
                    out.append(_text(buf))
                    buf = []
                out.append(f"⟨{ID_TO_SPECIAL.get(t, '?')}⟩")
        if buf:
            out.append(_text(buf))
        return "".join(out)

    def encode_text(self, text: str) -> list[int]:
        return _bytes(text)


def _find(ids: list[int], target: int, start: int) -> int:
    for k in range(start, len(ids)):
        if ids[k] == target:
            return k
    return -1


def _text(ids: Sequence[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")


def _parse_call(raw: str) -> ParsedToolCall:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return ParsedToolCall(name=None, arguments=None, raw=raw, ok=False)
    if (
        not isinstance(obj, dict)
        or not isinstance(obj.get("name"), str)
        or not isinstance(obj.get("arguments", {}), dict)
    ):
        return ParsedToolCall(name=None, arguments=None, raw=raw, ok=False)
    return ParsedToolCall(name=obj["name"], arguments=obj.get("arguments", {}), raw=raw, ok=True)
