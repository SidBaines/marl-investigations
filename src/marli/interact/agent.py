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
