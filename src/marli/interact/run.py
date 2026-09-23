"""Run one episode of one protocol on one task — the single entry point used by
eval rollouts, RL rollouts and tests.

::

    @dataclass
    class EpisodeSpec:
        protocol: Protocol
        env: Env                        # already constructed for this task
        task: Task
        seating: dict[str, str]         # role -> policy_id
        policies: dict[str, Policy]     # policy_id -> TokenPolicy | ChatPolicy
        # policy_id -> renderer FACTORY (token policies only): renderers are per-agent-lineage
        # objects (they remember the tool specs they rendered), so each agent gets its own.
        renderers: dict[str, Callable[[], DeltaRenderer]]
        limits: Limits
        schedule: str = "lockstep"      # lockstep | async
        delivery: DeliverySpec = DeliverySpec()
        run_seed: int = 0
        episode_idx: int = 0
        group_id: str = ""
        config_hash: str = ""
        protocol_name: str = ""
        backend: str = ""
        clock: Clock | None = None

    async def run_episode(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        # 1. env.setup(); build Workspace, Scheduler, Ledger, Recorder, SystemIO
        # 2. outcome = await protocol.run(io)   (bounded by limits.episode.max_wall_s)
        # 3. grades: per-agent (grade_individual) + "_system" for outcome.final_answer
        # 4. metrics = compute_metrics(...); assemble Episode; env.teardown() in finally
        # Never raises for agent/backend failures: returns ok=False with errors.
        # Raises ConfigError for invalid specs (unseated role, renderer/policy mismatch,
        # trainable seat with non-raw sampling, ctx > backend max_seq_len).
"""

from __future__ import annotations
