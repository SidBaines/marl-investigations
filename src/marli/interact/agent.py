"""AgentRuntime — one agent instance's loop over its context lineage.

This is where the token-exactness invariant is enforced: the runtime owns the
agent's append-only segment buffer, appends sampled ids verbatim, and renders
only deltas. Everything else (tools, workspace, scheduler, limits, context
managers, policies) is injected.

Loop (token policies; ``supports_delta=True``)::

    seg = pending_segment(START | SPAWN | SESSION)  # recorded lazily at the first call
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
        if last_term is not None and context.should_compact(len(buf)):
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
        ledger.charge(...); parsed = renderer.parse(completion_ids)
        last_term = "stop" if completion_ids[-1] in renderer.stop_token_ids else "length"
        tool_msgs = []
        for call in parsed.tool_calls (in order):
            # shared tools run in tool_phase; release it before a blocking tool
            run tool -> tool message (truncated); stop after a control tool
        apply control effects: submit -> done(submission); end_session -> session boundary;
                               return_report -> done(report)
        no tool calls -> Limits.on_no_tool_call (nudge adds a user message next turn; end_agent;
        final_text_as_answer; final_text_continue retains a provisional answer)

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
* Backends must raise BackendError for transport/API failures after retries.
  These end the agent with
  ``ended_by="error"`` and mark the episode not-ok; they never crash the
  episode's other agents. Other exceptions propagate, including spend/config
  errors and programming errors.
* A segment is fresh until its first completion. Its recorder buffer is
  created at the first call, so harness instructions belong in initial(),
  before the generation header. Fresh segments never compact; an oversized
  initial prompt follows on_exhaust, ending with "ctx" if even FINAL cannot fit.
* COMPACT and CARRY reset context and proceed to ACT in the same ticket.
  Context reserves protect the final instruction and prefix; on ctx exhaustion
  pending observations are omitted from that final call to preserve its space.
* System prompt n_agents is the role's count, or 0 for dynamic roles. Budget
  fields use the protocol-adjusted limits documented by RoleSpec.

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
from contextlib import AsyncExitStack
from copy import copy
from dataclasses import dataclass, field, replace
from typing import Any, cast

from marli.errors import BackendError, ConfigError
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

    def assistant(self, parsed: ParsedTurn, call_id: str) -> None:
        self.messages.append(
            Msg(
                "assistant",
                parsed.content,
                tool_calls=tuple(
                    ToolCall(
                        call.name or "unparsed",
                        call.arguments or {},
                        call.id or f"{call_id}/t{index}",
                    )
                    for index, call in enumerate(parsed.tool_calls)
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
    """Resolve advertised tools, preferring trusted environment tools to built-ins.

    Overrides retain the built-in's turn-ending control semantics and must return
    its control effect, e.g. ``control={"submit": source}`` to end the agent.
    Each name is advertised once, even when the role lists it more than once.
    """
    available = {tool.spec.name: tool for tool in tools}
    names = list(role.tools)
    if context.has_notes_tools:
        names.extend(["read_notes", "write_notes"])
    resolved: list[Tool] = []
    for name in dict.fromkeys(names):
        if name in available:
            tool = available[name]
            if name in TOOLS.names() and not tool.control and TOOLS.get(name)().control:
                tool = copy(tool)
                # Env tools may be frozen records; alter only this role's copy.
                object.__setattr__(tool, "control", True)
            resolved.append(tool)
        elif name in TOOLS.names():
            resolved.append(TOOLS.get(name)())
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
            agent_id=info.agent_id,
            role=info.role,
            tick=None,
            seq=0,
            workspace=workspace,
            sandbox=sandbox,
            scheduler=scheduler,
            system=system,
            ledger=ledger,
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
            n_agents=role.count or 0,
            max_workers_per_call=limits.spawn.max_per_call,
            max_workers_total=limits.spawn.max_total,
            worker_tokens=limits.worker.max_gen_tokens,
            session_tokens=limits.session.max_gen_tokens,
        )
        self.log = MessageLog()
        self.segment_id = ""
        self._segment_index = 0
        self._reason = SegmentStart.SPAWN if info.parent else SegmentStart.START
        self._carry_from: str | None = None
        self._carry = ""
        self._fresh = True
        self._ctx_reserve = 0
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
                segment_id=self.segment_id,
                agent_id=self.info.agent_id,
                session_idx=self._session,
                start_reason=reason,
                carry_from=self._carry_from,
                renderer=self.renderer.name,
                tokenizer_sha=self.renderer.tokenizer_sha,
            ),
            ids,
        )

    def _initial(self, delivery: str | None) -> None:
        text = "\n\n".join(part for part in (self.first_message, self._carry, delivery) if part)
        sessions = self.limits.session.max_sessions
        if sessions > 1:
            label = f"[Session {self._session + 1} of {sessions}]"
            if self._session + 1 == sessions:
                label += " — final session: you must submit before it ends"
            text = label + "\n\n" + text
        self.log = MessageLog([Msg("user", text)])
        if self.renderer and self.limits.on_exhaust == "force_final":
            # An empty assistant measures the unsampled close and the next header
            # without asking continuation() to follow a nonexistent completion.
            base = self.renderer.initial(self.system_prompt, self.tool_specs, self.log.messages)
            final = self.renderer.initial(
                self.system_prompt,
                self.tool_specs,
                self.log.messages + [Msg("assistant", ""), Msg("user", self._final_instruction)],
            )
            self._ctx_reserve = (
                getattr(self.limits, self.role.limits_key).final_reserve
                + len(final)
                - len(base)
                + len(
                    self.renderer.suppress_thinking_prefix()
                    if self._text_final
                    else self.renderer.forced_tool_prefix(self._final_tool)
                )
            )
        self._last_term = "stop"

    @property
    def _final_tool(self) -> str:
        return "return_report" if self.role.limits_key == "worker" else "submit"

    @property
    def _text_final(self) -> bool:
        return (
            self.limits.on_no_tool_call == "final_text_continue"
            and self._final_tool not in self.tools
        )

    @property
    def _final_instruction(self) -> str:
        if self._session_pending and self._session + 1 == self.limits.session.max_sessions:
            return "This is your last session. Submit your final answer now."
        return (
            "You have run out of budget. Return your report now."
            if self.role.limits_key == "worker"
            else "You have run out of budget. Submit your final answer now."
        )

    def _reset(self, reason: SegmentStart, summary: str | None = None) -> None:
        notes = None
        if self.context.has_notes_tools:
            # A control tool can follow a notes write in the same lockstep turn.
            view = self.workspace.view(self.info.agent_id)
            notes = view.own_staged.get("notes")
            if notes is None:
                notes, _ = self.workspace.read(self.info.agent_id, self.info.agent_id, "notes")
        carry = self.context.carry_text(
            summary=summary, notes=notes, session=reason == SegmentStart.SESSION
        )
        if carry.truncated:
            self.ledger.limit_hit(self.info.agent_id, "context.carry_truncated")
        self._carry = carry.text
        if reason == SegmentStart.SESSION and self.context.spec.kind == "tail" and self.renderer:
            tail = self.context.tail_text(
                [
                    call.completion_ids
                    for call in self.recorder.calls_of(self.info.agent_id)
                    if call.session_idx == self._session
                ],
                self.renderer,
            )
            if tail:
                self._carry = "\n\n".join(part for part in (self._carry, tail) if part)
        self._carry_from = self.segment_id or None
        self.segment_id = ""
        self._reason = reason
        self._fresh = True
        self._tool_messages = []
        self._reads = []
        self._nudges = []
        self._n_nudges = 0  # a fresh context starts a fresh nudge budget
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
        if self._fresh or not self.renderer.supports_delta:
            return self.renderer.initial(
                self.system_prompt,
                self.tool_specs,
                self.log.messages + messages,
            )
        return self.renderer.continuation(self._last_term, messages)

    def _prompt_len(self, delta: list[int]) -> int:
        if self.renderer is None:
            return 0
        return len(delta) + (
            len(self._buffer()) if self.renderer.supports_delta and not self._fresh else 0
        )

    def _allocate(self, ticket: Ticket, length: int, purpose: Purpose) -> Allocation:
        return self.ledger.allocate(
            self.info.agent_id,
            prompt_len=length,
            purpose=purpose,
            tick=ticket.tick,
            n_active=self.ledger.n_active,
            ctx_reserve=self._ctx_reserve,
        )

    async def run(self) -> AgentResult:
        try:
            await self._loop()
            return AgentResult(
                self.info.agent_id,
                self.submission,
                self.report,
                self.ended_by,
                self._session + 1,
            )
        except asyncio.CancelledError:
            if "episode.max_wall_s" in self.ledger.limits_hit().get("_episode", ()):
                self.ended_by = "max_wall_s"
            raise
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
            delta = self._prompt(messages)
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
            compact = not fresh and self.context.should_compact(self._prompt_len(delta))
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
                            [],
                            delivered_reads if fresh else [],
                            forced=not self._session_pending,
                        )
                        if self.error:
                            return
                        summary = parsed.content if parsed else None
                    self._reset(SegmentStart.SESSION, summary)
                    self._initial(delivery)
                    compact = False
                    messages, reads = [], delivered_reads
                    delta = self._prompt([])
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
                    self._initial(delivery)
                    messages, reads = [], delivered_reads
                    delta = self._prompt([])
                    allocation = self._allocate(ticket, self._prompt_len(delta), Purpose.ACT)
                elif self.error:
                    return
                else:
                    allocation = Allocation(0, "ctx.max_ctx")
            if allocation.exhausted:
                self.ended_by = {
                    "episode.max_ticks": "max_ticks",
                    "ctx.max_ctx": "ctx",
                }.get(allocation.exhausted, "budget")
                if (
                    self.limits.on_exhaust == "force_final"
                    and self.submission is None
                    and self.report is None
                ):
                    worker = self.role.limits_key == "worker"
                    if self.ended_by == "ctx":
                        messages = []
                        reads = reads if self._fresh else []
                    await self._harness_call(
                        ticket,
                        Purpose.REPORT if worker else Purpose.FINAL,
                        self._final_instruction,
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
            if self.error is not None or parsed is None or self._apply_control(control):
                return
            if parsed.tool_calls:
                self._n_nudges = 0
            elif self.limits.on_no_tool_call == "final_text_as_answer":
                if "return_report" in self.tools:
                    self.report, self.ended_by = parsed.content, "report"
                else:
                    self.submission, self.ended_by = parsed.content, "submit"
                return
            elif self.limits.on_no_tool_call == "final_text_continue":
                self.submission = parsed.content
            elif self.limits.on_no_tool_call == "end_agent":
                return
            elif self._n_nudges >= self.limits.max_nudges:
                self.ledger.limit_hit(self.info.agent_id, "max_nudges")
                self.ended_by = "max_nudges"
                if (
                    self.limits.on_exhaust == "force_final"
                    and self.submission is None
                    and self.report is None
                ):
                    worker = self.role.limits_key == "worker"
                    await self._harness_call(
                        ticket,
                        Purpose.REPORT if worker else Purpose.FINAL,
                        self._final_instruction,
                        [],
                        [],
                        forced=True,
                    )
                return
            else:
                # A turn cut off by its token allocation did not decline the
                # tools; it only gets the nudge message, not a strike.
                if parsed.termination != Termination.LENGTH:
                    self._n_nudges += 1
                if "return_report" in self.tools:
                    nudge = "Please use a tool or call return_report with your findings."
                elif "submit" in self.tools:
                    nudge = "Please use a tool to act or submit your final answer."
                elif "end_session" in self.tools:
                    nudge = "Please use a tool to act, or call end_session when you are done."
                else:
                    nudge = "Please use a tool to act."
                self._nudges = [Msg("user", nudge)]

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
        if not messages and self.log.messages and self.log.messages[-1].tool_calls:
            # Dropping pending results must also drop their declarations from
            # semantic history. The sampled token buffer remains untouched.
            self.log.messages[-1] = replace(self.log.messages[-1], tool_calls=())
        messages = messages + [Msg("user", instruction)]
        delta = self._prompt(messages)
        prefix: list[int] = []
        if self.renderer:
            # Forced tool prefixes already skip reasoning in the format's native way
            # (Qwen3.5 closes the prefilled think block; Harmony opens the commentary
            # channel), so only summary calls use suppress_thinking_prefix — adding both
            # would emit e.g. a second </think>.
            if purpose in (Purpose.FINAL, Purpose.REPORT) and not self._text_final:
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
        ended_by = self.ended_by
        self._apply_control(control)
        if parsed and self._text_final and purpose == Purpose.FINAL and not parsed.tool_calls:
            self.submission = parsed.content
        if forced and purpose in (Purpose.FINAL, Purpose.REPORT) and not self.error:
            self.ended_by = ended_by
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
            if self.renderer.supports_delta and not self._fresh:
                self.recorder.extend(self.segment_id, delta + prefix)
            else:
                self._carry_from = self.segment_id or self._carry_from
                reason = self._reason if self.renderer.supports_delta else SegmentStart.RERENDER
                self._new_segment(delta + prefix, reason)
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
        except BackendError as exc:
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
            self._fresh = False
            self._last_term = (
                ("stop" if ids and ids[-1] in self.renderer.stop_token_ids else "length")
                if self.renderer
                else parsed.termination
            )
            self.log.assistant(parsed, call_id)
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
                    call_id=call_id,
                    episode_id=self.recorder.episode_id,
                    agent_id=self.info.agent_id,
                    role=self.info.role,
                    policy_id=self.info.policy_id,
                    policy_version=version,
                    segment_id=self.segment_id,
                    prompt_len=prompt_len,
                    completion_ids=ids,
                    logprobs=logprobs,
                    text=text,
                    termination=Termination(parsed.termination) if parsed else Termination.ERROR,
                    purpose=purpose,
                    forced=forced,
                    tool_calls=tuple(tool_records),
                    reads=tuple(reads),
                    tick=ticket.tick,
                    seq=seq,
                    session_idx=self._session,
                    seed=seed,
                    usage=usage,
                    timing=timing,
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
        stopped = False
        async with AsyncExitStack() as phase:
            if shared:
                await phase.enter_async_context(self.scheduler.tool_phase(self.info.agent_id))
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
                    agent_id=self.info.agent_id,
                    role=self.info.role,
                    tick=ticket.tick,
                    seq=seq,
                    workspace=self.workspace,
                    sandbox=self.sandbox,
                    scheduler=self.scheduler,
                    system=self.system,
                    ledger=self.ledger,
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
                    # Async joins must release the episode mutex. Lockstep has
                    # a single ordered phase per ticket: block() retires it when
                    # workers start, while a rejected spawn retains seat order.
                    if getattr(tool, "blocking", False) and self.scheduler.kind == "async":
                        await phase.aclose()
                        shared = False
                    elif tool.shared and not shared:
                        await phase.enter_async_context(
                            self.scheduler.tool_phase(self.info.agent_id)
                        )
                        shared = True
                    try:
                        result = await run_tool(tool, ctx, cast(dict[str, Any], call.arguments))
                    except BackendError as exc:
                        self.error = f"{self.info.agent_id}: {type(exc).__name__}: {exc}"
                        self.ended_by = "error"
                        result = ToolResult(f"error: {self.error}", self.error)
                    stopped = self.error is not None or (tool.control and result.error is None)
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
                if not shared:
                    await phase.enter_async_context(self.scheduler.tool_phase(self.info.agent_id))
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
