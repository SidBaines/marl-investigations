"""Built-ins expose validated calls and auditable effects without hiding runtime bugs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from marli.errors import ConfigError
from marli.interact import tools as tools_module
from marli.interact.records import Recorder
from marli.interact.scheduler import FakeClock, LockstepScheduler
from marli.interact.tools import (
    TOOLS,
    ToolCtx,
    ToolError,
    ToolResult,
    run_tool,
    truncate_output,
    validate_tool_spec,
)
from marli.interact.types import ReadVia, WorkspaceRead
from marli.interact.workspace import DeliverySpec, Permissions, Workspace
from marli.render.base import ToolSpec


@dataclass
class FakeScheduler:
    kind: str = "async"
    changed: bool = True
    waits: list[tuple[str, float]] = field(default_factory=list)

    async def wait_for_update(self, agent_id: str, timeout: float) -> bool:
        self.waits.append((agent_id, timeout))
        return self.changed


@pytest.fixture
def ctx() -> ToolCtx:
    return ToolCtx(
        agent_id="alice",
        role="peer",
        tick=3,
        seq=7,
        workspace=Workspace(
            roles={"alice": "peer", "bob": "peer"},
            permissions={"peer": Permissions(notes=True)},
            delivery=DeliverySpec(),
            staged=False,
        ),
        sandbox=None,
        scheduler=FakeScheduler(),
        system=None,
        ledger=None,
    )


@pytest.mark.parametrize(
    ("name", "shared", "control"),
    [
        ("write_scratchpad", True, False),
        ("read_scratchpad", False, False),
        ("list_scratchpads", False, False),
        ("read_notes", False, False),
        ("write_notes", True, False),
        ("submit", False, True),
        ("end_session", False, True),
        ("return_report", False, True),
        ("wait_for_update", False, False),
    ],
)
def test_factories_and_tool_metadata(name: str, shared: bool, control: bool) -> None:
    tool = TOOLS.get(name)()
    assert isinstance(tool.spec, ToolSpec)
    assert tool.spec.name == name
    assert tool.spec.description
    assert tool.spec.parameters["type"] == "object"
    assert tool.spec.parameters["additionalProperties"] is False
    assert tool.shared is shared
    assert tool.control is control


def test_factories_take_no_context_and_isolate_mutable_specs(ctx: ToolCtx) -> None:
    factory = TOOLS.get("write_scratchpad")
    with pytest.raises(TypeError):
        factory(ctx)
    first, second = factory(), factory()
    first.spec.parameters["properties"]["mode"]["enum"].append("insert")
    first.spec.parameters["required"].append("mode")
    for tool in (second, factory()):
        assert tool.spec.parameters["properties"]["mode"]["enum"] == ["append", "overwrite"]
        assert tool.spec.parameters["required"] == ["content"]


@pytest.mark.parametrize("staged", [False, True])
async def test_write_scratchpad_defaults_append_and_overwrite(ctx: ToolCtx, staged: bool) -> None:
    ctx.workspace.staged = staged
    tool = TOOLS.get("write_scratchpad")()
    suffix = " (visible to others next tick)" if staged else ""
    assert await run_tool(tool, ctx, {"content": "one"}) == ToolResult(
        "ok: scratchpad v1 (3 chars)" + suffix
    )
    assert await run_tool(tool, ctx, {"content": "two"}) == ToolResult(
        "ok: scratchpad v2 (7 chars)" + suffix
    )
    assert await run_tool(tool, ctx, {"content": "new", "mode": "overwrite"}) == ToolResult(
        "ok: scratchpad v3 (3 chars)" + suffix
    )
    if staged:
        assert ctx.workspace.log() == ()
        assert ctx.workspace.view("alice").own_staged == {"scratchpad": "new"}
        ctx.workspace.commit_staged(["alice"])
    assert [write.content for write in ctx.workspace.log()] == ["one", "one\ntwo", "new"]
    assert all(write.tick == 3 and write.seq == 7 for write in ctx.workspace.log())


async def test_read_scratchpad_empty_latest_and_selected_version(ctx: ToolCtx) -> None:
    tool = TOOLS.get("read_scratchpad")()
    assert await run_tool(tool, ctx, {"agent_id": "bob"}) == ToolResult(
        "(empty)", control={"reads": [WorkspaceRead("bob", "scratchpad", 0, ReadVia.PULL)]}
    )
    ctx.workspace.write("bob", "scratchpad", "one", mode="overwrite", tick=0, seq=1)
    ctx.workspace.write("bob", "scratchpad", "two", mode="overwrite", tick=0, seq=2)
    for version, content in [(None, "two"), (0, "(empty)"), (1, "one"), (2, "two")]:
        arguments = {"agent_id": "bob"}
        if version is not None:
            arguments["version"] = version
        assert await run_tool(tool, ctx, arguments) == ToolResult(
            content,
            control={
                "reads": [
                    WorkspaceRead(
                        "bob", "scratchpad", 2 if version is None else version, ReadVia.PULL
                    )
                ]
            },
        )
    result = await run_tool(tool, ctx, {"agent_id": "bob", "version": 3})
    assert result.error and "no version 3" in result.error


async def test_list_scratchpads(ctx: ToolCtx) -> None:
    ctx.workspace.add_agent("empty", "peer")
    ctx.workspace.write("alice", "scratchpad", "mine", mode="overwrite", tick=0, seq=0)
    ctx.workspace.write("bob", "scratchpad", "one\ntwo", mode="overwrite", tick=0, seq=1)
    result = await run_tool(TOOLS.get("list_scratchpads")(), ctx, {})
    assert result == ToolResult(
        "alice v1 (4 chars): mine\nbob v1 (7 chars): one\nempty v0 (0 chars): ",
        control={"reads": [WorkspaceRead("bob", "scratchpad", 1, ReadVia.PULL)]},
    )


async def test_own_writes_can_be_read_within_one_lockstep_tick(ctx: ToolCtx) -> None:
    ctx.workspace.staged = True
    ctx.scheduler = LockstepScheduler(ctx.workspace, Recorder("test", clock=FakeClock()))
    ctx.scheduler.register("alice", (0,))
    ticket = await ctx.scheduler.turn("alice")
    ctx.tick = ticket.tick
    assert ctx.tick == 0
    async with ctx.scheduler.tool_phase("alice"):
        for key in ("notes", "scratchpad"):
            write = TOOLS.get(f"write_{key}")()
            read = TOOLS.get(f"read_{key}")()
            arguments = {} if key == "notes" else {"agent_id": "alice"}
            for content in ("first", "second"):
                assert (
                    await run_tool(write, ctx, {"content": content, "mode": "overwrite"})
                ).error is None
                result = await run_tool(read, ctx, arguments)
                assert result.content == content
                assert result.error is None
                assert not result.control.get("reads")
            if key == "scratchpad":
                result = await run_tool(read, ctx, {"agent_id": "alice", "version": 2})
                assert result.content == "second"
                assert result.control["reads"] == []
        assert ctx.workspace.log() == ()
        assert ctx.workspace.read("bob", "alice") == ("", 0)
    ctx.scheduler.done("alice")
    assert len(ctx.workspace.log()) == 4
    assert ctx.workspace.read("bob", "alice") == ("second", 2)


async def test_own_committed_scratchpad_records_no_cross_read(ctx: ToolCtx) -> None:
    ctx.workspace.write("alice", "scratchpad", "mine", mode="overwrite", tick=0, seq=1)
    result = await run_tool(TOOLS.get("read_scratchpad")(), ctx, {"agent_id": "alice"})
    assert result == ToolResult("mine", control={"reads": []})


async def test_read_write_notes_and_overwrite_default(ctx: ToolCtx) -> None:
    read_tool, write_tool = TOOLS.get("read_notes")(), TOOLS.get("write_notes")()
    assert await run_tool(read_tool, ctx, {}) == ToolResult("(no notes yet)")
    assert await run_tool(write_tool, ctx, {"content": "old"}) == ToolResult(
        "ok: notes v1 (3 chars)"
    )
    assert await run_tool(write_tool, ctx, {"content": "new"}) == ToolResult(
        "ok: notes v2 (3 chars)"
    )
    assert await run_tool(write_tool, ctx, {"content": "tail", "mode": "append"}) == ToolResult(
        "ok: notes v3 (8 chars)"
    )
    assert await run_tool(read_tool, ctx, {}) == ToolResult("new\ntail")


@pytest.mark.parametrize("staged", [False, True])
async def test_notes_cap_checks_total_content_and_failures_leave_no_writes(
    ctx: ToolCtx, staged: bool
) -> None:
    ctx.workspace = Workspace(
        roles={"alice": "peer"},
        permissions={"peer": Permissions(notes=True)},
        delivery=DeliverySpec(),
        staged=staged,
        notes_cap_chars=5,
    )
    tool = TOOLS.get("write_notes")()
    assert (await run_tool(tool, ctx, {"content": "12345"})).error is None
    for arguments in [
        {"content": "123456"},
        {"content": "x", "mode": "append"},
        {"content": "", "mode": "append"},
    ]:
        result = await run_tool(tool, ctx, arguments)
        assert result.error and "5 chars" in result.error
        assert result.content == f"error: {result.error}"
    if staged:
        assert ctx.workspace.view("alice").own_staged == {"notes": "12345"}
        ctx.workspace.commit_staged(["alice"])
    assert len(ctx.workspace.log()) == 1
    assert ctx.workspace.read("alice", "alice", "notes") == ("12345", 1)
    assert (await run_tool(tool, ctx, {"content": "x"})).error is None
    ctx.workspace.commit_staged(["alice"])
    assert ctx.workspace.read("alice", "alice", "notes") == ("x", 2)


@pytest.mark.parametrize(
    ("name", "arguments", "content", "effect"),
    [
        ("submit", {"answer": "42"}, "submitted", {"submit": "42"}),
        ("end_session", {}, "session ended", {"end_session": True}),
        ("return_report", {"report": "found it"}, "report returned", {"report": "found it"}),
    ],
)
async def test_control_effects(
    ctx: ToolCtx, name: str, arguments: dict[str, Any], content: str, effect: dict[str, Any]
) -> None:
    assert await run_tool(TOOLS.get(name)(), ctx, arguments) == ToolResult(content, control=effect)
    assert ctx.workspace.log() == ()


@pytest.mark.parametrize(
    ("changed", "text"), [(True, "workspace updated"), (False, "no update (timed out)")]
)
@pytest.mark.parametrize("timeout", [None, 0, 1.5, 300])
async def test_wait_for_update(
    ctx: ToolCtx, changed: bool, text: str, timeout: float | None
) -> None:
    ctx.scheduler.changed = changed
    arguments = {} if timeout is None else {"timeout_s": timeout}
    assert await run_tool(TOOLS.get("wait_for_update")(), ctx, arguments) == ToolResult(text)
    assert ctx.scheduler.waits == [("alice", 30 if timeout is None else timeout)]


async def test_wait_for_update_rejects_lockstep_without_waiting(ctx: ToolCtx) -> None:
    ctx.scheduler.kind = "lockstep"
    with pytest.raises(ConfigError, match="async scheduler"):
        await run_tool(TOOLS.get("wait_for_update")(), ctx, {})
    assert ctx.scheduler.waits == []


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("write_scratchpad", {}, "missing required arguments"),
        ("write_scratchpad", {"content": 7}, "must be string"),
        ("write_scratchpad", {"content": "x", "mode": "insert"}, "must be one of"),
        ("write_scratchpad", {"content": "x", "extra": True}, "unknown arguments"),
        ("read_scratchpad", {}, "missing required arguments"),
        ("read_scratchpad", {"agent_id": None}, "must be string"),
        ("read_scratchpad", {"agent_id": "bob", "version": True}, "must be integer"),
        ("read_scratchpad", {"agent_id": "bob", "version": 1.0}, "must be integer"),
        ("read_scratchpad", {"agent_id": "bob", "version": "1"}, "must be integer"),
        ("read_scratchpad", {"agent_id": "bob", "version": None}, "must be integer"),
        ("read_scratchpad", {"agent_id": "bob", "version": -1}, "at least 0"),
        ("list_scratchpads", {"agent_id": "bob"}, "unknown arguments"),
        ("read_notes", {"agent_id": "bob"}, "unknown arguments"),
        ("write_notes", {}, "missing required arguments"),
        ("write_notes", {"content": []}, "must be string"),
        ("write_notes", {"content": "x", "mode": "insert"}, "must be one of"),
        ("submit", {}, "missing required arguments"),
        ("submit", {"answer": 42}, "must be string"),
        ("end_session", {"answer": "x"}, "unknown arguments"),
        ("return_report", {}, "missing required arguments"),
        ("return_report", {"report": {}}, "must be string"),
        ("wait_for_update", {"timeout_s": True}, "must be number"),
        ("wait_for_update", {"timeout_s": "30"}, "must be number"),
        ("wait_for_update", {"timeout_s": -1}, "at least 0"),
        ("wait_for_update", {"timeout_s": 301}, "at most 300"),
        ("wait_for_update", {"timeout_s": float("nan")}, "finite"),
        ("wait_for_update", {"timeout_s": float("inf")}, "finite"),
    ],
)
async def test_invalid_arguments_become_errors_before_effects(
    ctx: ToolCtx, name: str, arguments: dict[str, Any], message: str
) -> None:
    result = await run_tool(TOOLS.get(name)(), ctx, arguments)
    assert result.error and message in result.error
    assert result.content == f"error: {result.error}"
    assert result.control == {}
    assert ctx.workspace.log() == ()
    assert ctx.scheduler.waits == []


@pytest.mark.parametrize("arguments", [None, [], "{}", 3])
async def test_arguments_must_be_json_object(ctx: ToolCtx, arguments: Any) -> None:
    result = await run_tool(TOOLS.get("end_session")(), ctx, arguments)
    assert result.error == "arguments must be a JSON object"
    assert result.control == {}


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("read_scratchpad", {"agent_id": "bob"}, "may not read"),
        ("read_scratchpad", {"agent_id": "missing"}, "unknown agent"),
        ("write_scratchpad", {"content": "x"}, "may not write scratchpad"),
        ("write_notes", {"content": "x"}, "may not write notes"),
    ],
)
async def test_workspace_tool_errors_are_user_visible(
    ctx: ToolCtx, name: str, arguments: dict[str, Any], message: str
) -> None:
    ctx.workspace = Workspace(
        roles={"alice": "peer", "bob": "peer"},
        permissions={"peer": Permissions(read_others=False, write_scratchpad=False)},
        delivery=DeliverySpec(),
        staged=False,
    )
    result = await run_tool(TOOLS.get(name)(), ctx, arguments)
    assert result.error and message in result.error
    assert result.content == f"error: {result.error}"


class FailingTool:
    spec = ToolSpec("failure", "Exercise error handling.", {"type": "object", "properties": {}})
    shared = False
    control = False

    def __init__(self, error: Exception) -> None:
        validate_tool_spec(self.spec)
        self.error = error

    async def __call__(self, ctx: ToolCtx, **arguments: Any) -> ToolResult:
        raise self.error


async def test_only_tool_errors_are_converted(ctx: ToolCtx) -> None:
    assert await run_tool(FailingTool(ToolError("expected")), ctx, {}) == ToolResult(
        "error: expected", error="expected"
    )
    error = RuntimeError("bug")
    with pytest.raises(RuntimeError, match="bug") as caught:
        await run_tool(FailingTool(error), ctx, {})
    assert caught.value is error


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("read_scratchpad", {"agent_id": "bob"}),
        ("read_notes", {}),
        ("list_scratchpads", {}),
        ("write_scratchpad", {"content": "x"}),
        ("write_notes", {"content": "x"}),
    ],
)
async def test_unknown_harness_agent_is_not_a_tool_error(
    ctx: ToolCtx, name: str, arguments: dict[str, Any]
) -> None:
    ctx.agent_id = "missing"
    with pytest.raises(ValueError, match="unknown agent"):
        await run_tool(TOOLS.get(name)(), ctx, arguments)


@pytest.mark.parametrize(
    ("property_schema", "construct"),
    [
        ({"type": ["string", "null"]}, "type"),
        ({"type": "unknown"}, "unknown"),
        ({"type": None}, "type"),
        ({"$ref": "#/definitions/value"}, "$ref"),
        ({"type": "string", "pattern": "x"}, "pattern"),
        ({"anyOf": [{"type": "string"}]}, "anyOf"),
        ({"type": "array", "items": {"type": "string"}}, "items"),
        ({"type": "object", "properties": {}}, "properties"),
        ({"enum": "abc"}, "enum"),
        ({"enum": []}, "enum"),
        ({"enum": [{"nested": True}]}, "enum"),
        ({"enum": [float("nan")]}, "enum"),
        ({"type": "string", "minimum": 0}, "minimum"),
        ({"minimum": 0}, "minimum"),
        ({"type": "number", "maximum": True}, "maximum"),
        ({"type": "number", "maximum": float("inf")}, "maximum"),
        ({"type": "number", "minimum": 2, "maximum": 1}, "minimum"),
        (False, "schema"),
    ],
)
def test_unsupported_schema_fails_at_tool_construction(
    monkeypatch: pytest.MonkeyPatch, property_schema: Any, construct: str
) -> None:
    def factory() -> FailingTool:
        tool = FailingTool(RuntimeError("must not execute"))
        tool.spec = ToolSpec(
            "custom", "Custom schema", {"type": "object", "properties": {"value": property_schema}}
        )
        return tool

    monkeypatch.setitem(TOOLS._functions, "custom", factory)
    with pytest.raises(ConfigError, match=re.escape(construct)):
        TOOLS.get("custom")()


@pytest.mark.parametrize(
    ("schema", "construct"),
    [
        (False, "schema"),
        ({"type": "array"}, "type"),
        ({"type": ["object", "null"]}, "type"),
        ({"type": "object", "$ref": "#"}, "$ref"),
        ({"type": "object", "additionalProperties": True}, "additionalProperties"),
        ({"type": "object", "properties": []}, "properties"),
        ({"type": "object", "required": "value"}, "required"),
        ({"type": "object", "required": ["missing"]}, "required"),
        ({"type": "object", "required": [False]}, "required"),
    ],
)
def test_invalid_root_schema_is_configuration_error(schema: Any, construct: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(construct)):
        validate_tool_spec(ToolSpec("custom", "Custom schema", schema))


async def test_schema_checked_once_per_built_tool(
    monkeypatch: pytest.MonkeyPatch, ctx: ToolCtx
) -> None:
    checked = []

    def check(spec: ToolSpec) -> None:
        checked.append(spec)
        validate_tool_spec(spec)

    monkeypatch.setattr(tools_module, "validate_tool_spec", check)
    factory = TOOLS.get("submit")
    first, second = factory(), factory()
    assert checked == [first.spec, second.spec]
    for _ in range(2):
        assert (await run_tool(first, ctx, {"answer": "42"})).error is None
        assert (await run_tool(second, ctx, {"answer": "42"})).error is None
    assert len(checked) == 2


@pytest.mark.parametrize("value", [True, False, 1.0, "1"])
async def test_enum_membership_preserves_types(ctx: ToolCtx, value: Any) -> None:
    tool = FailingTool(ToolError("handler reached"))
    tool.spec = ToolSpec(
        "enum", "Scalar enum", {"type": "object", "properties": {"n": {"enum": [1, 2]}}}
    )
    validate_tool_spec(tool.spec)
    result = await run_tool(tool, ctx, {"n": value})
    assert result.error and "must be one of" in result.error
    assert (await run_tool(tool, ctx, {"n": 1})).error == "handler reached"


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("string", "s"),
        ("integer", 1),
        ("number", 1.5),
        ("boolean", True),
        ("null", None),
        ("object", {"x": 1}),
        ("array", [1]),
    ],
)
async def test_supported_schema_types(ctx: ToolCtx, kind: str, value: Any) -> None:
    tool = FailingTool(ToolError("handler reached"))
    tool.spec = ToolSpec(
        "type", "Type check", {"type": "object", "properties": {"n": {"type": kind}}}
    )
    validate_tool_spec(tool.spec)
    assert (await run_tool(tool, ctx, {"n": value})).error == "handler reached"


@pytest.mark.parametrize(("text", "limit"), [("", 0), ("hello", 5), ("hello", 10)])
def test_short_output_unchanged(text: str, limit: int) -> None:
    assert truncate_output(text, limit) == text


def test_truncation_keeps_head_tail_and_counts_omitted_chars() -> None:
    text = "HEAD" + "x" * 92 + "TAIL"
    result = truncate_output(text, 50)
    assert len(result) == 50
    assert result.startswith("HEAD")
    assert result.endswith("TAIL")
    match = re.search(r"\n…\[(\d+) chars truncated\]…\n", result)
    assert match is not None
    assert int(match[1]) == len(text) - (len(result) - len(match[0]))
    head, tail = result[: match.start()], result[match.end() :]
    assert head == text[: len(head)]
    assert tail == text[-len(tail) :]
    assert abs(len(head) - len(tail)) <= 1


@pytest.mark.parametrize("length", [1, 10, 100, 1000])
def test_truncation_respects_all_limits_including_small_marker_budgets(length: int) -> None:
    text = "a" * length
    for limit in range(length + 1):
        result = truncate_output(text, limit)
        assert len(result) <= limit
        if length > limit and limit < len(f"\n…[{length} chars truncated]…\n"):
            assert result == f"\n…[{length} chars truncated]…\n"[:limit]


def test_negative_output_limit_is_configuration_error() -> None:
    with pytest.raises(ConfigError, match="non-negative"):
        truncate_output("hello", -1)
