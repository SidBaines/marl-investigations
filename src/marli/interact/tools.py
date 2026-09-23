"""Tools: specs the model sees + async handlers the harness runs.

A tool is registered by name (``TOOLS``) as a factory ``() -> Tool``;
roles list tool names (``RoleSpec.tools``). Environments contribute their own
tools (``python``, ``bash``) through ``Env.tools(role)``.

Tool calls in one turn run **in the order sampled**. A *control* tool
(``submit``, ``end_session``, ``return_report``, ``spawn_workers``) ends the agent's turn: any
later calls in the same turn receive an error result and are not executed.
Tools with ``shared=True`` (anything touching the shared sandbox or workspace
writes) run inside ``scheduler.tool_phase`` so lockstep ordering is
deterministic; pure tools (reads) may run concurrently. A ``blocking=True``
tool releases the async tool mutex so the agents it waits on can acquire it.
Lockstep's block/unblock gates hand the phase to workers without a mutex.

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
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import wraps
from typing import Any, Protocol

from marli.errors import ConfigError
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
    blocking: bool = False  # waits for other agents; must release the async tool mutex
    control: bool  # ends the turn (submit / end_session / return_report / spawn_workers)

    async def __call__(self, ctx: ToolCtx, **arguments: Any) -> ToolResult: ...


class _ToolRegistry(FnRegistry[Callable[[], Tool]]):
    """Validate schemas and isolate their mutable dictionaries at construction."""

    def get(self, ref: str) -> Callable[[], Tool]:
        factory = super().get(ref)

        @wraps(factory)
        def build() -> Tool:
            tool = factory()
            tool.spec = replace(tool.spec, parameters=deepcopy(tool.spec.parameters))
            validate_tool_spec(tool.spec)
            return tool

        return build


TOOLS: FnRegistry = _ToolRegistry("tools")  # name -> factory() -> Tool


_SCHEMA_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
    "object": (dict,),
    "array": (list,),
}


def validate_tool_spec(spec: ToolSpec) -> None:
    """Check the supported JSON-schema subset once, when a tool is built.

    TOOLS factories call this automatically; environment tool constructors
    must call it themselves. Schemas describe a top-level object with named
    properties, required names, and optional additionalProperties=False.
    Each property supports a single type (string, integer, number, boolean,
    null, object or array), a scalar enum, and numeric minimum/maximum.
    Object/array values are type-checked only; nested schemas are unsupported.
    Description/default annotations are accepted without applying defaults.
    All other constructs, including type lists and $ref, raise ConfigError.
    """
    schema = spec.parameters
    context = f"tool {spec.name!r} schema"
    if not isinstance(schema, dict):
        raise ConfigError(f"{context}: expected an object schema")
    unknown = schema.keys() - {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "description",
        "default",
    }
    if unknown:
        raise ConfigError(f"{context}: unsupported constructs {sorted(unknown, key=str)}")
    if schema.get("type") != "object":
        raise ConfigError(f"{context}: unsupported type {schema.get('type')!r}; expected object")
    if schema.get("additionalProperties", False) is not False:
        raise ConfigError(f"{context}: unsupported additionalProperties; expected false")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or any(not isinstance(key, str) for key in properties):
        raise ConfigError(f"{context}: properties must map names to schemas")
    required = schema.get("required", [])
    if (
        not isinstance(required, list)
        or any(not isinstance(name, str) for name in required)
        or len(set(required)) != len(required)
        or set(required) - properties.keys()
    ):
        raise ConfigError(f"{context}: required must list distinct declared property names")
    for name, prop in properties.items():
        location = f"{context} property {name!r}"
        if not isinstance(prop, dict):
            raise ConfigError(f"{location}: expected a property schema")
        unknown = prop.keys() - {"type", "enum", "minimum", "maximum", "description", "default"}
        if unknown:
            raise ConfigError(f"{location}: unsupported constructs {sorted(unknown, key=str)}")
        kind = prop.get("type")
        if "type" in prop and (not isinstance(kind, str) or kind not in _SCHEMA_TYPES):
            raise ConfigError(f"{location}: unsupported type {kind!r}")
        if "enum" in prop:
            choices = prop["enum"]
            if (
                not isinstance(choices, list)
                or not choices
                or any(
                    type(choice) not in (str, int, float, bool, type(None)) for choice in choices
                )
                or any(
                    isinstance(choice, float) and not math.isfinite(choice) for choice in choices
                )
            ):
                raise ConfigError(f"{location}: enum must be a non-empty list of JSON scalars")
        for bound in ("minimum", "maximum"):
            if bound in prop and (
                kind not in {"integer", "number"}
                or type(prop[bound]) not in (int, float)
                or (isinstance(prop[bound], float) and not math.isfinite(prop[bound]))
            ):
                raise ConfigError(f"{location}: {bound} requires a numeric type and finite bound")
        if "minimum" in prop and "maximum" in prop and prop["minimum"] > prop["maximum"]:
            raise ConfigError(f"{location}: minimum exceeds maximum")


async def run_tool(tool: Tool, ctx: ToolCtx, arguments: dict[str, Any]) -> ToolResult:
    """Validate arguments against a prechecked spec; convert only model-input errors.

    Tool construction must call validate_tool_spec (automatic via TOOLS).
    Schema checks happen once there, while every call's arguments are checked
    here against that supported JSON-schema subset.
    """
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
    for name, value in arguments.items():
        spec = properties[name]
        kind = spec.get("type")
        if kind is not None and type(value) not in _SCHEMA_TYPES[kind]:
            raise ToolError(f"argument {name!r} must be {kind}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolError(f"argument {name!r} must be finite")
        if "enum" in spec and not any(
            type(value) is type(choice) and value == choice for choice in spec["enum"]
        ):
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
        raise ConfigError("tool output limit must be non-negative")
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
    blocking = False
    control = False


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
        "Read a scratchpad, including your own staged writes, optionally at a version.",
        _parameters(
            {"agent_id": {"type": "string"}, "version": {"type": "integer", "minimum": 0}},
            ("agent_id",),
        ),
    )

    async def __call__(
        self, ctx: ToolCtx, *, agent_id: str, version: int | None = None
    ) -> ToolResult:
        content, actual_version = ctx.workspace.read(ctx.agent_id, agent_id, version=version)
        reads = (
            [WorkspaceRead(agent_id, "scratchpad", actual_version, ReadVia.PULL)]
            if agent_id != ctx.agent_id
            else []
        )
        return ToolResult(content or "(empty)", control={"reads": reads})


@TOOLS.register("list_scratchpads")
class _ListScratchpads(_BuiltinTool):
    spec = ToolSpec("list_scratchpads", "List readable scratchpads.", _parameters({}))

    async def __call__(self, ctx: ToolCtx) -> ToolResult:
        entries = [
            entry
            for entry in ctx.workspace.list_index(ctx.agent_id)
            if entry.writer != ctx.agent_id or entry.version > 0
        ]
        return ToolResult(
            "\n".join(
                f"{entry.writer} v{entry.version} ({entry.n_chars} chars): {entry.first_line}"
                for entry in entries
            ),
            control={
                "reads": [
                    WorkspaceRead(entry.writer, "scratchpad", entry.version, ReadVia.PULL)
                    for entry in entries
                    if entry.writer != ctx.agent_id and entry.version > 0
                ]
            },
        )


@TOOLS.register("read_notes")
class _ReadNotes(_BuiltinTool):
    spec = ToolSpec(
        "read_notes", "Read your private notes, including staged writes.", _parameters({})
    )

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
            raise ConfigError("wait_for_update requires an async scheduler")
        changed = await ctx.scheduler.wait_for_update(ctx.agent_id, timeout_s)
        return ToolResult("workspace updated" if changed else "no update (timed out)")
