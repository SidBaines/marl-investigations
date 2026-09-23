"""Tools: specs the model sees + async handlers the harness runs.

A tool is registered by name (``TOOLS``) as a factory ``(ToolCtx) -> Tool``;
roles list tool names (``RoleSpec.tools``). Environments contribute their own
tools (``python``, ``bash``) through ``Env.tools(role)``.

Tool calls in one turn run **in the order sampled**. A *control* tool
(``submit``, ``end_session``, ``return_report``) ends the agent's turn: any
later calls in the same turn receive an error result and are not executed.
Tools with ``shared=True`` (anything touching the shared sandbox or workspace
writes) run inside ``scheduler.tool_phase`` so lockstep ordering is
deterministic; pure tools (reads) may run concurrently.

Results are truncated (head + tail, ``Limits.tool_output_chars``) before they
enter the agent's context; the truncated text is what is recorded.

Built-in tools: ``write_scratchpad(content, mode="append"|"overwrite")``,
``read_scratchpad(agent_id, version=None)``, ``list_scratchpads()``,
``read_notes()``, ``write_notes(content, mode="overwrite")``, ``submit(answer)``,
``end_session()``, ``return_report(report)``, ``wait_for_update(timeout_s)``
(async only). ``spawn_workers(tasks=[{task, context}])`` is registered by the
coordinator protocol (M2-3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from marli.interact.types import ReadVia, WorkspaceRead
from marli.registry import FnRegistry
from marli.render.base import ToolSpec


class ToolError(Exception):
    """A user-visible tool failure: becomes an error result for the agent, not a crash."""


@dataclass(frozen=True)
class ToolResult:
    content: str
    error: str | None = None
    # Control effects the runtime applies after the tool returns, e.g.
    # {"submit": "42"} | {"end_session": True} | {"report": "..."}.
    control: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCtx:
    """Everything a handler may touch. Graders never get one."""

    agent_id: str
    role: str
    tick: int | None
    seq: int
    workspace: Any  # interact.workspace.Workspace
    sandbox: Any | None  # envs.sandbox.base.Sandbox (None for math-only envs)
    scheduler: Any  # interact.scheduler.Scheduler (for block/unblock in spawn)
    system: Any  # interact.system.SystemIO (spawn_workers starts agents through it)
    ledger: Any  # interact.limits.Ledger


class Tool(Protocol):
    spec: ToolSpec
    shared: bool
    control: bool  # ends the turn (submit / end_session / return_report)

    async def __call__(self, ctx: ToolCtx, **arguments: Any) -> ToolResult: ...


TOOLS: FnRegistry = FnRegistry("tools")  # name -> factory() -> Tool


async def run_tool(tool: Tool, ctx: ToolCtx, arguments: dict[str, Any]) -> ToolResult:
    """Validate a call and turn only user-visible ToolErrors into error results."""
    try:
        _validate_arguments(tool.spec.parameters, arguments)
        return await tool(ctx, **arguments)
    except ToolError as exc:
        message = str(exc)
        return ToolResult(content=f"error: {message}", error=message)


def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ToolError("arguments must be a JSON object")
    properties = schema.get("properties", {})
    unknown = arguments.keys() - properties.keys()
    if unknown:
        raise ToolError(f"unknown arguments: {sorted(unknown, key=str)}")
    missing = set(schema.get("required", [])) - arguments.keys()
    if missing:
        raise ToolError(f"missing required arguments: {sorted(missing)}")
    types = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "null": (type(None),),
        "object": (dict,),
        "array": (list,),
    }
    for name, value in arguments.items():
        spec = properties[name]
        kind = spec.get("type")
        if kind is not None and type(value) not in types[kind]:
            raise ToolError(f"argument {name!r} must be {kind}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolError(f"argument {name!r} must be finite")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"argument {name!r} must be one of {spec['enum']}")
        if "minimum" in spec and value < spec["minimum"]:
            raise ToolError(f"argument {name!r} must be at least {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            raise ToolError(f"argument {name!r} must be at most {spec['maximum']}")


def truncate_output(text: str, limit: int) -> str:
    """Fit head, omission count and tail within a character limit.

    When even the marker cannot fit, return its prefix. A zero limit returns
    an empty string; a negative limit is a configuration error.
    """
    if limit < 0:
        raise ValueError("tool output limit must be non-negative")
    if len(text) <= limit:
        return text
    kept = limit
    while True:
        marker = f"\n…[{len(text) - kept} chars truncated]…\n"
        available = max(0, limit - len(marker))
        if kept <= available:
            break
        kept = available
    head = (kept + 1) // 2
    tail = kept // 2
    return text[:head] + marker[:limit] + (text[-tail:] if tail else "")


def _parameters(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


class _BuiltinTool:
    shared = False
    control = False

    def __init__(self, ctx: ToolCtx | None = None) -> None:
        """Accept factory context; each invocation uses its current ToolCtx."""


@TOOLS.register("write_scratchpad")
class _WriteScratchpad(_BuiltinTool):
    shared = True
    spec = ToolSpec(
        "write_scratchpad",
        "Append to or replace your scratchpad.",
        _parameters(
            {
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["append", "overwrite"], "default": "append"},
            },
            ("content",),
        ),
    )

    async def __call__(self, ctx: ToolCtx, *, content: str, mode: str = "append") -> ToolResult:
        write = ctx.workspace.write(
            ctx.agent_id, "scratchpad", content, mode=mode, tick=ctx.tick, seq=ctx.seq
        )
        text = f"ok: scratchpad v{write.version} ({len(write.content)} chars)"
        if ctx.workspace.staged:
            text += " (visible to others next tick)"
        return ToolResult(text)


@TOOLS.register("read_scratchpad")
class _ReadScratchpad(_BuiltinTool):
    spec = ToolSpec(
        "read_scratchpad",
        "Read an agent's committed scratchpad, optionally at a specific version.",
        _parameters(
            {"agent_id": {"type": "string"}, "version": {"type": "integer", "minimum": 0}},
            ("agent_id",),
        ),
    )

    async def __call__(
        self, ctx: ToolCtx, *, agent_id: str, version: int | None = None
    ) -> ToolResult:
        content, actual_version = ctx.workspace.read(ctx.agent_id, agent_id, version=version)
        read = WorkspaceRead(agent_id, "scratchpad", actual_version, ReadVia.PULL)
        return ToolResult(content or "(empty)", control={"reads": [read]})


@TOOLS.register("list_scratchpads")
class _ListScratchpads(_BuiltinTool):
    spec = ToolSpec("list_scratchpads", "List readable scratchpads.", _parameters({}))

    async def __call__(self, ctx: ToolCtx) -> ToolResult:
        return ToolResult(
            "\n".join(
                f"{entry.writer} v{entry.version} ({entry.n_chars} chars): {entry.first_line}"
                for entry in ctx.workspace.list_index(ctx.agent_id)
            )
        )


@TOOLS.register("read_notes")
class _ReadNotes(_BuiltinTool):
    spec = ToolSpec("read_notes", "Read your private committed notes.", _parameters({}))

    async def __call__(self, ctx: ToolCtx) -> ToolResult:
        content, _ = ctx.workspace.read(ctx.agent_id, ctx.agent_id, "notes")
        return ToolResult(content or "(no notes yet)")


@TOOLS.register("write_notes")
class _WriteNotes(_BuiltinTool):
    shared = True
    spec = ToolSpec(
        "write_notes",
        "Replace or append to your private notes, within the workspace notes cap.",
        _parameters(
            {
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["append", "overwrite"], "default": "overwrite"},
            },
            ("content",),
        ),
    )

    async def __call__(self, ctx: ToolCtx, *, content: str, mode: str = "overwrite") -> ToolResult:
        write = ctx.workspace.write(
            ctx.agent_id, "notes", content, mode=mode, tick=ctx.tick, seq=ctx.seq
        )
        return ToolResult(f"ok: notes v{write.version} ({len(write.content)} chars)")


@TOOLS.register("submit")
class _Submit(_BuiltinTool):
    control = True
    spec = ToolSpec(
        "submit",
        "Submit your final answer.",
        _parameters({"answer": {"type": "string"}}, ("answer",)),
    )

    async def __call__(self, ctx: ToolCtx, *, answer: str) -> ToolResult:
        return ToolResult("submitted", control={"submit": answer})


@TOOLS.register("end_session")
class _EndSession(_BuiltinTool):
    control = True
    spec = ToolSpec("end_session", "End your current session.", _parameters({}))

    async def __call__(self, ctx: ToolCtx) -> ToolResult:
        return ToolResult("session ended", control={"end_session": True})


@TOOLS.register("return_report")
class _ReturnReport(_BuiltinTool):
    control = True
    spec = ToolSpec(
        "return_report",
        "Return your report to the coordinator.",
        _parameters({"report": {"type": "string"}}, ("report",)),
    )

    async def __call__(self, ctx: ToolCtx, *, report: str) -> ToolResult:
        return ToolResult("report returned", control={"report": report})


@TOOLS.register("wait_for_update")
class _WaitForUpdate(_BuiltinTool):
    spec = ToolSpec(
        "wait_for_update",
        "Wait for another agent's update (async scheduling only).",
        _parameters({"timeout_s": {"type": "number", "default": 30, "minimum": 0, "maximum": 300}}),
    )

    async def __call__(self, ctx: ToolCtx, *, timeout_s: float = 30) -> ToolResult:
        if ctx.scheduler.kind != "async":
            raise ToolError("wait_for_update requires an async scheduler")
        if not math.isfinite(timeout_s) or not 0 <= timeout_s <= 300:
            raise ToolError("timeout_s must be between 0 and 300 seconds")
        changed = await ctx.scheduler.wait_for_update(ctx.agent_id, timeout_s)
        return ToolResult("workspace updated" if changed else "no update (timed out)")
