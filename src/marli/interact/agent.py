"""AgentRuntime — one agent instance's loop over its context lineage.

This is where the token-exactness invariant is enforced: the runtime owns the
agent's append-only segment buffer, appends sampled ids verbatim, and renders
only deltas. Everything else (tools, workspace, scheduler, limits, context
managers, policies) is injected.

Loop (token policies; ``supports_delta=True``)::

    seg = new_segment(START | SPAWN | SESSION)
    buf = renderer.initial(system_prompt, tool_specs, [user(first_message + first_delivery)])
    last_term = None
    while True:
        ticket = await scheduler.turn(agent_id)
        if last_term is not None:                                    # not the first call of the
        segment
            delivery_text, reads = workspace.pending_delivery(agent_id)
            buf += renderer.continuation(last_term, tool_msgs +
            [user(delivery_text)]*bool(delivery_text)
                                         + nudge_msgs)
        if context.should_compact(len(buf)):                          # context.py: compaction/tail
            buf, seg = await context.compact(...)                     # purpose=COMPACT call, new
            segment
        alloc = ledger.allocate(agent_id, prompt_len=len(buf), ...)
        if alloc.exhausted:
            -> force_final / carry / stop per Limits.on_exhaust (see limits.py); record limits_hit
        sample = await policy.sample(buf, SamplingSpec(max_tokens=alloc.max_tokens,
                                                       stop_token_ids=renderer.stop_token_ids, ...),
                                                       seed=seed)
        record Call(prompt_len=len(buf), completion_ids=sample.completion_ids, ...); buf +=
        completion_ids
        ledger.charge(...); parsed = renderer.parse(completion_ids); last_term = parsed.termination
        tool_msgs = []
        async with scheduler.tool_phase(agent_id):  # only if any called tool is shared
            for call in parsed.tool_calls (in order):  run tool -> tool message (truncated); stop
            after a control tool
        apply control effects: submit -> done(submission); end_session -> session boundary;
                               return_report -> done(report)
        no tool calls -> Limits.on_no_tool_call (nudge adds a user message next turn; end_agent;
        final_text_as_answer)

Details that are part of the contract:

* ``Call.prompt_len`` is ``len(buf)`` at sampling time; the completion is
  appended at exactly that offset (so datums can locate every action token).
* Observation tokens: the initial prompt, every continuation, the prefilled
  ``forced_tool_prefix`` and forced-close tokens. Only sampled
  ``completion_ids`` are actions.
* ``supports_delta=False`` renderers: every call starts a fresh segment
  (``SegmentStart.RERENDER``) built by ``renderer.initial`` from the agent's
  MessageLog (assistant history rendered with thinking kept per renderer
  policy). Datums then come one per call — correct but O(T²) prefill.
* Chat (API) policies: no token buffer; the runtime keeps a MessageLog and
  calls ``policy.chat(messages, tools, cache_salt=seeds.cache_salt(...))``.
  Calls are recorded with ``segment_id=""``, ``prompt_len=0``,
  ``completion_ids=()``; usage is tagged with the API tokenizer.
* Seeds: ``derive_seed(run_seed, task_id, episode_idx, agent_id, call_index)``.
* Sessions (``SessionLimits.max_sessions > 1``): a session ends on
  ``end_session()`` or when the session budget is exhausted; the runtime then
  makes the carry call (context.py), starts a new SESSION segment
  ``initial(system, tools, [user(first_message + carry_text)])`` and
  continues. ``submit`` ends the whole agent.
* Errors from the policy backend (after its retries) end the agent with
  ``ended_by="error"`` and mark the episode not-ok; they never crash the
  episode's other agents.

Constructor (M1-8)::

    AgentRuntime(info: AgentInfo, *, policy, renderer: DeltaRenderer | None, role: RoleSpec,
                 tools: list[Tool], first_message: str, scheduler, workspace, ledger, limits:
                 Limits,
                 context: ContextManager, recorder: Recorder, sandbox, system, run_seed: int,
                 task_id: str, episode_idx: int, sampling: SamplingOverrides)
    async run() -> AgentResult
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Any, cast

from marli.errors import ConfigError
from marli.interact.context import ContextManager
from marli.interact.limits import Allocation, Ledger, Limits
from marli.interact.records import Recorder
from marli.interact.scheduler import Scheduler, Ticket
from marli.interact.system import AgentResult, RoleSpec, SystemIO
from marli.interact.tools import TOOLS, Tool, ToolCtx, ToolResult, run_tool, truncate_output
from marli.interact.types import (
    AgentInfo,
    Call,
    EventKind,
    Purpose,
    SegmentInfo,
    SegmentStart,
    Termination,
    Timing,
    ToolCallRecord,
    Usage,
    WorkspaceRead,
)
from marli.interact.workspace import Workspace
from marli.policy.base import (
    CallMeta,
    ChatPolicy,
    Policy,
    SamplingSpec,
    TokenPolicy,
    check_trainable_sampling,
)
from marli.render.base import DeltaRenderer, Msg, ParsedTurn, ToolCall
from marli.seeds import cache_salt, derive_seed


@dataclass(frozen=True)
class SamplingOverrides:
    """Per-seat sampling controls; allocations always determine the token cap."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1


@dataclass
class MessageLog:
    """Semantic history for chat seats and renderers that require a full prompt."""

    messages: list[Msg] = field(default_factory=list)

    def assistant(self, parsed: ParsedTurn) -> None:
        self.messages.append(
            Msg(
                "assistant",
                parsed.content,
                tool_calls=tuple(
                    ToolCall(call.name, call.arguments, call.id)
                    for call in parsed.tool_calls
                    if call.ok and call.name is not None and call.arguments is not None
                ),
                thinking=parsed.thinking,
            )
        )


def role_tools(
    role: RoleSpec,
    tools: Sequence[Tool],
    context: ContextManager,
    ctx: ToolCtx,
) -> list[Tool]:
    """Resolve only advertised environment tools, plus the role's built-ins."""
    available = {tool.spec.name: tool for tool in tools}
    names = list(role.tools)
    if context.has_notes_tools:
        names.extend(["read_notes", "write_notes"])
    resolved: list[Tool] = []
    for name in dict.fromkeys(names):
        if name in TOOLS.names():
            resolved.append(TOOLS.get(name)())
        elif name in available:
            resolved.append(available[name])
        else:
            raise ConfigError(f"role {role.role!r}: unknown tool {name!r}")
    return resolved


class AgentRuntime:
    """Own one context lineage; sampling is the only source of action tokens."""

    def __init__(
        self,
        info: AgentInfo,
        *,
        policy: Policy,
        renderer: DeltaRenderer | None,
        role: RoleSpec,
        tools: list[Tool],
        first_message: str,
        scheduler: Scheduler,
        workspace: Workspace,
        ledger: Ledger,
        limits: Limits,
        context: ContextManager,
        recorder: Recorder,
        sandbox: Any,
        system: SystemIO,
        run_seed: int,
        task_id: str,
        episode_idx: int,
        sampling: SamplingOverrides,
    ) -> None:
        self.info = info
        self.policy = policy
        self.renderer = renderer
        self.role = role
        self.scheduler = scheduler
        self.workspace = workspace
        self.ledger = ledger
        self.limits = limits
        self.context = context
        self.recorder = recorder
        self.sandbox = sandbox
        self.system = system
        self.first_message = first_message
        self.run_seed = run_seed
        self.task_id = task_id
        self.episode_idx = episode_idx
        self.sampling = sampling
        tool_ctx = ToolCtx(
            info.agent_id,
            info.role,
            None,
            0,
            workspace,
            sandbox,
            scheduler,
            system,
            ledger,
        )
        self.tools = {tool.spec.name: tool for tool in role_tools(role, tools, context, tool_ctx)}
        self.tool_specs = [tool.spec for tool in self.tools.values()]
        if isinstance(policy, TokenPolicy):
            if renderer is None or renderer.name != policy.renderer_name:
                raise ConfigError(
                    f"policy {info.policy_id!r}: renderer mismatch or missing renderer"
                )
        elif policy.trainable:
            raise ConfigError("trainable chat policy is not supported")
        if policy.trainable:
            check_trainable_sampling(self._sampling_spec(1), policy_id=info.policy_id)
        self.system_prompt = role.system_prompt.format(
            agent_id=info.agent_id,
            role=info.role,
            n_agents=ledger.expected_agents,
        )
        self.log = MessageLog()
        self.segment_id = ""
        self._segment_index = 0
        self._reason = SegmentStart.SPAWN if info.parent else SegmentStart.START
        self._carry_from: str | None = None
        self._carry = ""
        self._tail: list[int] = []
        self._fresh = True
        self._last_term = "stop"
        self._tool_messages: list[Msg] = []
        self._reads: list[WorkspaceRead] = []
        self._nudges: list[Msg] = []
        self._n_nudges = 0
        self._call_index = 0
        self._session = 0
        self._session_pending = False
        self.submission: str | None = None
        self.report: str | None = None
        self.error: str | None = None
        self.ended_by = "end_agent"
        self._stop_reason: str | None = None

    def request_stop(self, reason: str) -> None:
        """Let an in-flight turn finish; honor the first stop at the next turn."""
        if self._stop_reason is None:
            self._stop_reason = reason

    def _sampling_spec(self, max_tokens: int) -> SamplingSpec:
        return SamplingSpec(
            max_tokens,
            self.sampling.temperature,
            self.sampling.top_p,
            self.sampling.top_k,
            self.renderer.stop_token_ids if self.renderer else (),
        )

    def _buffer(self) -> list[int]:
        return self.recorder.buffer(self.segment_id) if self.segment_id else []

    def _new_segment(self, ids: list[int], reason: SegmentStart) -> None:
        assert self.renderer is not None
        self.segment_id = f"{self.info.agent_id}/g{self._segment_index}"
        self._segment_index += 1
        self.recorder.new_segment(
            SegmentInfo(
                self.segment_id,
                self.info.agent_id,
                self._session,
                reason,
                self._carry_from,
                self.renderer.name,
                self.renderer.tokenizer_sha,
            ),
            ids,
        )

    def _initial(self, delivery: str | None) -> None:
        text = "\n\n".join(part for part in (self.first_message, self._carry, delivery) if part)
        self.log = MessageLog([Msg("user", text)])
        if self.renderer and self.renderer.supports_delta:
            ids = self.renderer.initial(self.system_prompt, self.tool_specs, self.log.messages)
            self._new_segment(ids + self._tail, self._reason)
        self._fresh = False
        self._last_term = "stop"

    def _reset(self, reason: SegmentStart, summary: str | None = None) -> None:
        notes = None
        if self.context.has_notes_tools:
            # A control tool can follow a notes write in the same lockstep turn.
            view = self.workspace.view(self.info.agent_id)
            notes = view.own_staged.get("notes")
            if notes is None:
                notes, _ = self.workspace.read(self.info.agent_id, self.info.agent_id, "notes")
        carry = self.context.carry_text(summary=summary, notes=notes)
        if carry.truncated:
            self.ledger.limit_hit(self.info.agent_id, "context.carry_truncated")
        self._carry = carry.text
        self._tail = self.context.tail_prefill(self._buffer())
        self._carry_from = self.segment_id or None
        self._reason = reason
        self._fresh = True
        self._tool_messages = []
        self._reads = []
        self._nudges = []
        if reason == SegmentStart.SESSION:
            self._session += 1
            self.ledger.start_session(self.info.agent_id)
        self._session_pending = False

    def _observations(self, delivery: str | None) -> list[Msg]:
        messages = list(self._tool_messages)
        if delivery:
            mode = self.renderer.delivery_role if self.renderer else "user"
            if not messages or mode == "user":
                messages.append(Msg("user", delivery))
            elif mode == "append_tool":
                messages[-1] = replace(
                    messages[-1], content=messages[-1].content + "\n\n" + delivery
                )
            else:
                messages.append(Msg("tool", delivery, name="workspace"))
        return messages + self._nudges

    def _prompt(self, messages: list[Msg]) -> list[int]:
        if self.renderer is None:
            return []
        if not self.renderer.supports_delta:
            return (
                self.renderer.initial(
                    self.system_prompt,
                    self.tool_specs,
                    self.log.messages + messages,
                )
                + self._tail
            )
        return self.renderer.continuation(self._last_term, messages)

    def _prompt_len(self, delta: list[int]) -> int:
        if self.renderer is None:
            return 0
        return len(delta) + (len(self._buffer()) if self.renderer.supports_delta else 0)

    def _allocate(self, ticket: Ticket, length: int, purpose: Purpose) -> Allocation:
        return self.ledger.allocate(
            self.info.agent_id,
            prompt_len=length,
            purpose=purpose,
            tick=ticket.tick,
            n_active=self.ledger.n_active,
        )

    async def run(self) -> AgentResult:
        try:
            await self._loop()
            if self.role.limits_key == "worker" and self.report is None and self.error is None:
                self.report = f"[worker {self.info.agent_id}: no report]"
            return AgentResult(
                self.info.agent_id,
                self.submission,
                self.report,
                self.ended_by,
                self._session + 1,
            )
        finally:
            self.recorder.add_event(
                EventKind.DONE,
                self.info.agent_id,
                self.scheduler.tick,
                ended_by=self.ended_by,
                error=self.error,
            )
            self.ledger.done(self.info.agent_id)
            self.ledger.release_worker(self.info.agent_id)
            self.scheduler.done(self.info.agent_id)

    async def _loop(self) -> None:
        while True:
            ticket = await self.scheduler.turn(self.info.agent_id)
            delivery, delivered_reads = self.workspace.pending_delivery(self.info.agent_id)
            if delivery:
                self.recorder.add_event(EventKind.PUSH, self.info.agent_id, ticket.tick)
            fresh = self._fresh
            if fresh:
                self._initial(delivery)
            messages = [] if fresh else self._observations(delivery)
            reads = self._reads + delivered_reads
            delta = (
                []
                if fresh and self.renderer and self.renderer.supports_delta
                else self._prompt(messages)
            )
            if self._stop_reason is not None:
                if self.submission is None and self.limits.on_exhaust == "force_final":
                    await self._harness_call(
                        ticket,
                        Purpose.FINAL,
                        "The protocol has stopped this agent. Submit your final answer now.",
                        messages,
                        reads,
                        forced=True,
                    )
                if self.error is None:
                    self.ended_by = self._stop_reason
                return
            compact = self.context.should_compact(self._prompt_len(delta))
            # A discarded delta cannot exhaust context before compaction gets a chance.
            allocation = self._allocate(
                ticket,
                0 if compact else self._prompt_len(delta),
                Purpose.ACT,
            )
            session_end = self._session_pending or allocation.exhausted == "session.max_gen_tokens"
            if session_end and self.limits.session.max_sessions > 1:
                if self._session + 1 < self.limits.session.max_sessions:
                    instruction = self.context.session_carry_instruction()
                    summary = None
                    if instruction is not None:
                        parsed = await self._harness_call(
                            ticket,
                            Purpose.CARRY,
                            instruction,
                            messages,
                            reads,
                            forced=not self._session_pending,
                        )
                        if self.error:
                            return
                        summary = parsed.content if parsed else None
                    self._reset(SegmentStart.SESSION, summary)
                    # One scheduler ticket gates at most one call.
                    if instruction is not None:
                        continue
                    self._initial(delivery)
                    compact = False
                    messages, reads = [], delivered_reads
                    delta = (
                        [] if self.renderer and self.renderer.supports_delta else self._prompt([])
                    )
                    allocation = self._allocate(ticket, self._prompt_len(delta), Purpose.ACT)
                else:
                    self.ledger.limit_hit(self.info.agent_id, "session.max_sessions")
                    allocation = Allocation(0, "session.max_sessions")
            if compact and allocation.exhausted is None:
                parsed = await self._harness_call(
                    ticket,
                    Purpose.COMPACT,
                    self.context.compaction_instruction(),
                    [],
                    reads if fresh else [],
                    forced=False,
                )
                if parsed is not None:
                    self._reset(SegmentStart.COMPACTION, parsed.content)
                    continue
                if self.error:
                    return
                allocation = Allocation(0, "ctx.max_ctx")
            if allocation.exhausted:
                self.ended_by = (
                    "max_ticks" if allocation.exhausted == "episode.max_ticks" else "budget"
                )
                if self.limits.on_exhaust == "force_final":
                    worker = self.role.limits_key == "worker"
                    instruction = (
                        "You have run out of budget. Return your report now."
                        if worker
                        else "You have run out of budget. Submit your final answer now."
                    )
                    await self._harness_call(
                        ticket,
                        Purpose.REPORT if worker else Purpose.FINAL,
                        instruction,
                        messages,
                        reads,
                        forced=True,
                    )
                return
            parsed, control = await self._call(
                ticket,
                Purpose.ACT,
                messages,
                delta,
                [],
                reads,
                allocation,
                forced=False,
            )
            if parsed is None or self._apply_control(control):
                return
            if parsed.tool_calls:
                self._n_nudges = 0
            elif self.limits.on_no_tool_call == "final_text_as_answer":
                self.submission, self.ended_by = parsed.content, "submit"
                return
            elif self.limits.on_no_tool_call == "end_agent":
                return
            elif self._n_nudges >= self.limits.max_nudges:
                self.ledger.limit_hit(self.info.agent_id, "max_nudges")
                return
            else:
                self._n_nudges += 1
                self._nudges = [
                    Msg("user", "Please use a tool to act or submit your final answer.")
                ]

    def _apply_control(self, control: dict[str, Any]) -> bool:
        if "submit" in control:
            self.submission = control["submit"]
            self.ended_by = "submit"
            return True
        if "report" in control:
            self.report = control["report"]
            self.ended_by = "report"
            return True
        if control.get("end_session"):
            if self.limits.session.max_sessions == 1:
                self.ended_by = "end_agent"
                return True
            self._session_pending = True
        return False

    async def _harness_call(
        self,
        ticket: Ticket,
        purpose: Purpose,
        instruction: str,
        messages: list[Msg],
        reads: list[WorkspaceRead],
        *,
        forced: bool,
    ) -> ParsedTurn | None:
        messages = messages + [Msg("user", instruction)]
        delta = self._prompt(messages)
        prefix: list[int] = []
        if self.renderer:
            # Forced tool prefixes already skip reasoning in the format's native way
            # (Qwen3.5 closes the prefilled think block; Harmony opens the commentary
            # channel), so only summary calls use suppress_thinking_prefix — adding both
            # would emit e.g. a second </think>.
            if purpose in (Purpose.FINAL, Purpose.REPORT):
                prefix = self.renderer.forced_tool_prefix(
                    "submit" if purpose == Purpose.FINAL else "return_report",
                )
            else:
                prefix = self.renderer.suppress_thinking_prefix()
        allocation = self._allocate(ticket, self._prompt_len(delta) + len(prefix), purpose)
        if allocation.exhausted:
            return None
        if purpose == Purpose.COMPACT:
            allocation = Allocation(min(allocation.max_tokens, self.context.spec.compact_reserve))
        parsed, control = await self._call(
            ticket,
            purpose,
            messages,
            delta,
            prefix,
            reads,
            allocation,
            forced=forced,
        )
        self._apply_control(control)
        return parsed

    async def _call(
        self,
        ticket: Ticket,
        purpose: Purpose,
        messages: list[Msg],
        delta: list[int],
        prefix: list[int],
        reads: list[WorkspaceRead],
        allocation: Allocation,
        *,
        forced: bool,
    ) -> tuple[ParsedTurn | None, dict[str, Any]]:
        self.log.messages.extend(messages)
        if self.renderer:
            if self.renderer.supports_delta:
                self.recorder.extend(self.segment_id, delta + prefix)
            else:
                self._carry_from = self.segment_id or self._carry_from
                self._new_segment(delta + prefix, SegmentStart.RERENDER)
        self._tail = []
        self._tool_messages, self._reads, self._nudges = [], [], []
        prompt_len = len(self._buffer())
        call_id = f"{self.recorder.episode_id}/{self.info.agent_id}/c{self._call_index}"
        meta = CallMeta(
            self.recorder.episode_id,
            self.info.agent_id,
            self.info.role,
            self._call_index,
            purpose.value,
        )
        seed_parts = (
            self.run_seed,
            self.task_id,
            self.episode_idx,
            self.info.agent_id,
            self._call_index,
        )
        seed = derive_seed(*seed_parts)
        versions = [
            [writer, key, version]
            for (writer, key), version in sorted(ticket.view.versions.items())
        ]
        seq = self.recorder.add_event(
            EventKind.CALL_START,
            self.info.agent_id,
            ticket.tick,
            call_id=call_id,
            versions=versions,
        )
        started = self.recorder.clock.now()
        ids: tuple[int, ...] = ()
        logprobs: tuple[float, ...] | None = None
        version: int | None = None
        usage = Usage()
        text = ""
        parsed: ParsedTurn | None = None
        cancelled: asyncio.CancelledError | None = None
        try:
            if self.renderer:
                sample = await cast(TokenPolicy, self.policy).sample(
                    tuple(self._buffer()),
                    self._sampling_spec(allocation.max_tokens),
                    seed=seed,
                    meta=meta,
                )
            else:
                reply = await cast(ChatPolicy, self.policy).chat(
                    [Msg("system", self.system_prompt), *self.log.messages],
                    self.tool_specs,
                    max_tokens=allocation.max_tokens,
                    temperature=self.sampling.temperature,
                    seed=seed,
                    cache_salt=cache_salt(*seed_parts),
                    meta=meta,
                )
        except asyncio.CancelledError as exc:
            cancelled = exc
            self.ended_by = "budget"
        except (
            TypeError,
            ValueError,
            LookupError,
            AttributeError,
            AssertionError,
            NotImplementedError,
        ):
            raise
        except Exception as exc:
            self.error = f"{self.info.agent_id}: {type(exc).__name__}: {exc}"
            self.ended_by = "error"
        else:
            if self.renderer:
                ids, logprobs = sample.completion_ids, sample.logprobs
                version, usage = sample.policy_version, sample.usage
                self.recorder.extend(self.segment_id, list(ids))
                parsed = self.renderer.parse(prefix + list(ids), self.tool_specs)
                text = self.renderer.decode(ids)
            else:
                usage, text = reply.usage, reply.content
                parsed = ParsedTurn(
                    reply.content, reply.thinking, reply.tool_calls, reply.termination
                )
            self.ledger.charge(
                self.info.agent_id,
                gen_tokens=len(ids) if self.renderer else usage.completion_tokens,
                purpose=purpose,
            )
            self._last_term = parsed.termination
            self.log.assistant(parsed)
        self._call_index += 1
        self.recorder.add_event(
            EventKind.CALL_END,
            self.info.agent_id,
            ticket.tick,
            call_id=call_id,
            error=self.error,
        )
        timing = Timing(started, self.recorder.clock.now() - started)
        tool_records: list[ToolCallRecord] = []
        control: dict[str, Any] = {}
        try:
            if parsed and purpose not in (Purpose.COMPACT, Purpose.CARRY):
                tool_records, control = await self._execute_tools(
                    parsed,
                    ticket,
                    call_id,
                    publish_seq=seq
                    if purpose == Purpose.ACT and self.role.publish_final_text and parsed.content
                    else None,
                )
        finally:
            self.recorder.add_call(
                Call(
                    call_id,
                    self.recorder.episode_id,
                    self.info.agent_id,
                    self.info.role,
                    self.info.policy_id,
                    version,
                    self.segment_id,
                    prompt_len,
                    ids,
                    logprobs,
                    text,
                    Termination(parsed.termination) if parsed else Termination.ERROR,
                    purpose,
                    forced,
                    tuple(tool_records),
                    tuple(reads),
                    ticket.tick,
                    seq,
                    self._session,
                    seed,
                    usage,
                    timing,
                )
            )
        if cancelled is not None:
            raise cancelled
        return parsed, control

    async def _execute_tools(
        self,
        parsed: ParsedTurn,
        ticket: Ticket,
        call_id: str,
        *,
        publish_seq: int | None = None,
    ) -> tuple[list[ToolCallRecord], dict[str, Any]]:
        records: list[ToolCallRecord] = []
        control: dict[str, Any] = {}
        shared = publish_seq is not None or any(
            call.ok and call.name in self.tools and self.tools[call.name].shared
            for call in parsed.tool_calls
        )
        phase = self.scheduler.tool_phase(self.info.agent_id) if shared else nullcontext()
        stopped = False
        async with phase:
            for index, call in enumerate(parsed.tool_calls):
                tool = self.tools.get(call.name) if call.ok else None
                seq = self.recorder.add_event(
                    EventKind.TOOL_START,
                    self.info.agent_id,
                    ticket.tick,
                    call_id=call_id,
                    index=index,
                    name=call.name,
                )
                ctx = ToolCtx(
                    self.info.agent_id,
                    self.info.role,
                    ticket.tick,
                    seq,
                    self.workspace,
                    self.sandbox,
                    self.scheduler,
                    self.system,
                    self.ledger,
                )
                before = len(self.workspace.log())
                if stopped:
                    result = ToolResult(
                        "error: tool call after control tool", "tool call after control tool"
                    )
                elif not call.ok:
                    result = ToolResult(
                        "error: could not parse tool call", "could not parse tool call"
                    )
                elif tool is None:
                    result = ToolResult(
                        f"error: unknown tool {call.name}", f"unknown tool {call.name}"
                    )
                else:
                    result = await run_tool(tool, ctx, cast(dict[str, Any], call.arguments))
                    stopped = tool.control
                writes = [
                    write
                    for write in self.workspace.log()[before:]
                    if write.writer == self.info.agent_id
                ]
                for write in writes:
                    self.recorder.add_write(write)
                    self.recorder.add_event(
                        EventKind.COMMIT,
                        write.writer,
                        ticket.tick,
                        key=write.key,
                        version=write.version,
                    )
                if writes or (self.workspace.staged and tool is not None and tool.shared):
                    # Lockstep notifications are a no-op; its scheduler records staged writes.
                    self.scheduler.notify_commit(self.info.agent_id)
                content = truncate_output(result.content, self.limits.tool_output_chars)
                self._tool_messages.append(
                    Msg(
                        "tool",
                        content,
                        name=call.name or "unparsed",
                        tool_call_id=call.id or f"{call_id}/t{index}",
                    )
                )
                self._reads.extend(result.control.get("reads", []))
                control.update(
                    {key: value for key, value in result.control.items() if key != "reads"}
                )
                records.append(
                    ToolCallRecord(
                        index,
                        call.name,
                        call.arguments,
                        call.raw,
                        call.ok,
                        content,
                        result.error,
                        tool.shared if tool else False,
                    )
                )
                self.recorder.add_event(
                    EventKind.TOOL_END,
                    self.info.agent_id,
                    ticket.tick,
                    call_id=call_id,
                    index=index,
                    error=result.error,
                )
            if publish_seq is not None:
                write = self.workspace.write(
                    self.info.agent_id,
                    "scratchpad",
                    parsed.content,
                    mode="overwrite",
                    tick=ticket.tick,
                    seq=publish_seq,
                )
                if not self.workspace.staged:
                    self.recorder.add_write(write)
                    self.recorder.add_event(
                        EventKind.COMMIT,
                        write.writer,
                        ticket.tick,
                        key=write.key,
                        version=write.version,
                    )
                self.scheduler.notify_commit(self.info.agent_id)
        return records, control
