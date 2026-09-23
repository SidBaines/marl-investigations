"""Keep rollout, credit and optimization on one synchronous policy version.

Only a saved step is a recovery boundary. Intermediate manifests live under
``checkpoints/`` because RunDir reserves the root manifest for completion.
Resume discards rollouts and metrics after that boundary and repeats the work;
interrupted/unsaved steps lose their compute, but their recorded spend remains
charged. Backend state and published samplers are never interchanged.
The prescribed task_id/step group identity requires distinct tasks within a
batch; an epoch wrap that repeats a task fails instead of merging credit groups.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from statistics import fmean
from typing import Any

from marli import runlog
from marli.budget import SpendGuard, tinker_cost
from marli.config import to_dict
from marli.envs.base import Task
from marli.envs.registry import make_env
from marli.errors import BudgetExceededError, ConfigError, HashMismatchError
from marli.eval.policies import PolicySpec, build_policies, resolve_spec
from marli.eval.policies import SamplingOverrides as FrozenSampling
from marli.handles import InputRef, atomic_write_text
from marli.interact import records
from marli.interact.agent import SamplingOverrides
from marli.interact.configs import build_protocol, resolve_protocol
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.types import Episode
from marli.model import ModelSpec, load_model
from marli.policy.base import Policy
from marli.policy.refs import parse_ref
from marli.render.base import DeltaRenderer
from marli.render.registry import get_renderer
from marli.rundir import RunDir
from marli.seeds import derive_seed
from marli.tasks.taskset import TaskSet, read_tasks
from marli.train.backends.base import Learner, SamplerSnapshot, TrainBackend
from marli.train.backends.registry import make_backend
from marli.train.checkpoint import Checkpoint, LearnerCheckpoint
from marli.train.credit import CreditContext, assign_credit, validate_credit
from marli.train.datums import build_datums, datum_stats
from marli.train.rl import TrainRLConfig
from marli.train.types import TrainDatum, seat_learner


@dataclass(frozen=True)
class DataCursor:
    epoch: int = 0
    index: int = 0


def take_tasks(
    tasks: Sequence[Task], cursor: DataCursor, count: int, *, seed: int
) -> tuple[list[Task], DataCursor]:
    """Reconstruct epoch permutations from a cursor without persisting RNG state."""
    if not tasks:
        raise ConfigError("training TaskSet must not be empty")
    epoch, index = cursor.epoch, cursor.index
    batch: list[Task] = []
    while len(batch) < count:
        order = list(tasks)
        random.Random(derive_seed(seed, "data", epoch)).shuffle(order)
        n = min(count - len(batch), len(tasks) - index)
        batch.extend(order[index : index + n])
        index += n
        if index == len(tasks):
            epoch, index = epoch + 1, 0
    return batch, DataCursor(epoch, index)


def _preflight(
    cfg: TrainRLConfig,
) -> tuple[CreditContext, dict[str, ModelSpec], dict[str, PolicySpec]]:
    cfg.__post_init__()
    name, _, protocol_config = resolve_protocol(cfg.protocol, cfg.protocol_config)
    roles = {role.role: role for role in build_protocol(cfg.protocol, cfg.protocol_config).roles()}
    if missing := roles.keys() - cfg.seating.keys():
        raise ConfigError(f"unseated protocol roles: {sorted(missing)}")
    if extra := cfg.seating.keys() - roles.keys():
        raise ConfigError(f"seating has unknown roles: {sorted(extra)}")
    if not cfg.learners:
        raise ConfigError("training requires at least one learner")
    frozen: dict[str, PolicySpec] = {}
    seated = set()
    for role, seat in cfg.seating.items():
        learner = seat_learner(seat)
        if learner is not None:
            if learner not in cfg.learners:
                raise ConfigError(f"role {role!r} references unknown learner {learner!r}")
            seated.add(learner)
        else:
            ref = parse_ref(seat, resolve_paths=True)
            if ref.kind == "ckpt":
                from marli.train.checkpoint import resolve_checkpoint_ref

                ref = parse_ref(resolve_checkpoint_ref(ref, None))
            frozen[role] = PolicySpec(
                str(ref), sampling=cfg.frozen_sampling.get(role, FrozenSampling())
            )
            _, model, _ = resolve_spec(frozen[role])
            if model and cfg.limits.ctx.max_ctx > model.max_ctx:
                raise ConfigError(f"ctx.max_ctx exceeds frozen model {model.name!r} max_ctx")
    if not cfg.allow_idle and (idle := cfg.learners.keys() - seated):
        raise ConfigError(f"idle learners have no seated role: {sorted(idle)}; set allow_idle")
    if invalid := cfg.frozen_sampling.keys() - frozen.keys():
        raise ConfigError(f"frozen_sampling requires frozen roles: {sorted(invalid)}")
    ctx = CreditContext(
        roles,
        cfg.seating,
        protocol_hash=hashlib.sha256(
            json.dumps([name, protocol_config], sort_keys=True).encode()
        ).hexdigest(),
    )
    for message in validate_credit(
        cfg.credit, ctx, group_size=cfg.group_size, allow_idle=cfg.allow_idle
    ):
        warnings.warn(message, stacklevel=2)
    models = {name: load_model(spec.base_model) for name, spec in cfg.learners.items()}
    for name, spec in cfg.learners.items():
        model = models[name]
        if spec.backend == "local":
            raise ConfigError("local training backend not implemented until M4")
        if spec.backend == "tinker" and (
            model.tinker_max_ctx is None or cfg.limits.ctx.max_ctx > model.tinker_max_ctx
        ):
            raise ConfigError(f"learner {name!r}: ctx.max_ctx exceeds model tinker_max_ctx")
        if cfg.limits.ctx.max_ctx > model.max_ctx:
            raise ConfigError(f"learner {name!r}: ctx.max_ctx exceeds model max_ctx")
    return ctx, models, frozen


def _rebase(checkpoint: Checkpoint, root: Path) -> Checkpoint:
    def records_at(records: dict[str, LearnerCheckpoint]) -> dict[str, LearnerCheckpoint]:
        result = {}
        for name, record in records.items():
            state = record["state"]
            if state and "://" not in state:
                state = os.path.relpath((checkpoint.root / state).resolve(), root)
            result[name] = {**record, "state": state}
        return result

    return replace(
        checkpoint,
        root=root,
        learners=records_at(checkpoint.learners),
        history=[
            {**saved, "learners": records_at(saved["learners"])} for saved in checkpoint.history
        ],
    )


def _latest_checkpoint(run: RunDir) -> Checkpoint | None:
    manifests = list(run.path("checkpoints").glob("step_*/checkpoint.json"))
    if not manifests:
        return None
    latest = Checkpoint.load(max(manifests, key=lambda p: int(p.parent.name[5:])))
    if latest.run_config_hash != run.config_hash or latest.config_hash != run.config_hash:
        raise HashMismatchError("training checkpoint config hash differs from run")
    return _rebase(latest, run.out)


async def _save_checkpoint(
    run: RunDir,
    learners: dict[str, Learner],
    samplers: dict[str, SamplerSnapshot],
    *,
    step: int,
    run_name: str,
    cursor: DataCursor,
    rae_state: dict[str, float],
    previous: Checkpoint | None,
    inputs: tuple[InputRef, ...],
    meta: dict[str, Any],
) -> Checkpoint:
    names = list(learners)
    states = await asyncio.gather(
        *(learners[name].save_state(f"{run_name}-{name}-s{step}-state") for name in names)
    )
    saved: dict[str, LearnerCheckpoint] = {}
    for name, state in zip(names, states, strict=True):
        learner = learners[name]
        snapshot = samplers.get(name)
        saved[name] = {
            "state": state if "://" in state else os.path.relpath(state, run.out),
            "sampler": snapshot.policy_ref if snapshot else None,
            "version": learner.version,
            "base_model": learner.spec.base_model,
            "backend": learner.spec.backend,
            "rank": learner.spec.rank,
        }
    history = list(previous.history) if previous else []
    # Step -1 is an emergency save of the initial state, not a completed update.
    if step >= 0:
        history.append({"step": step, "learners": saved})
    recovery = dict(previous.meta.get("history_state", {})) if previous else {}
    recovery[str(step)] = {
        "data_cursor": asdict(cursor),
        "rae_state": dict(rae_state),
        "sampler_paths": {name: snapshot.path for name, snapshot in samplers.items()},
        "spend": meta["spend"],
    }
    checkpoint = Checkpoint(
        root=run.out,
        config_hash=run.config_hash,
        run_config_hash=run.config_hash,
        step=step,
        learners=saved,
        rae_state=dict(rae_state),
        data_cursor=asdict(cursor),
        history=history,
        inputs=inputs,
        meta={
            **meta,
            "history_state": recovery,
            "sampler_paths": {name: s.path for name, s in samplers.items()},
        },
    )
    _rebase(checkpoint, run.path(f"checkpoints/step_{step:05d}")).save()
    return checkpoint


async def _rollouts(
    cfg: TrainRLConfig,
    run: RunDir,
    batch: Sequence[Task],
    *,
    step: int,
    policies: dict[str, Policy],
    renderers: dict[str, Callable[[], DeltaRenderer]],
    learners: dict[str, Learner],
    spend: SpendGuard,
) -> list[tuple[Episode, dict[str, list[int]]]]:
    prefix = f"rollouts/step_{step:05d}"
    for filename in ("episodes.jsonl", "tokens.jsonl"):
        atomic_write_text(run.path(f"{prefix}/{filename}"), "")
    # (slot, task, idx): the batch slot keeps group ids unique when an epoch wrap
    # (or batch_tasks > len(tasks)) puts the same task in one batch twice.
    pending = iter(
        (slot, task, idx) for slot, task in enumerate(batch) for idx in range(cfg.group_size)
    )
    results: list[tuple[Episode, dict[str, list[int]]]] = []
    stopped: Exception | None = None
    sampling = {
        role: SamplingOverrides(**to_dict(value)) for role, value in cfg.frozen_sampling.items()
    }
    protocol_name = resolve_protocol(cfg.protocol, cfg.protocol_config)[0]

    def append(rel: str, row: Mapping[str, Any]) -> None:
        run.append_row(f"{prefix}/{rel}", row)

    async def worker() -> None:
        nonlocal stopped
        while stopped is None:
            item = next(pending, None)
            if item is None:
                return
            slot, task, idx = item
            try:
                spend.check(0, "next episode")
                episode, buffers = await run_episode(
                    EpisodeSpec(
                        protocol=build_protocol(cfg.protocol, cfg.protocol_config),
                        env=make_env(cfg.env, cfg.env_config, task),
                        task=task,
                        seating={role: role for role in cfg.seating},
                        policies=policies,
                        renderers=renderers,
                        limits=cfg.limits,
                        schedule=cfg.schedule,
                        run_seed=derive_seed(cfg.seed, step),
                        episode_idx=idx,
                        group_id=f"{task.task_id}/s{step}/b{slot}",
                        config_hash=run.config_hash,
                        protocol_name=protocol_name,
                        backend=",".join(sorted({spec.backend for spec in cfg.learners.values()})),
                        sampling=sampling,
                    )
                )
                records.write_episode(append, episode, buffers, record_tokens=True)
                for call in episode.calls:
                    learner = seat_learner(cfg.seating[call.role])
                    if learner is not None:
                        assert call.policy_version == learners[learner].version, (
                            f"Call {call.call_id}: expected learner {learner!r} version "
                            f"{learners[learner].version}, got {call.policy_version}"
                        )
                results.append((episode, buffers))
                run.write_progress({"step": step - 1, "steps": cfg.steps, "spend": spend.summary()})
                spend.check(0, "completed episode")
            except Exception as exc:
                if stopped is None or isinstance(exc, BudgetExceededError):
                    stopped = exc
                return

    workers = [
        asyncio.create_task(worker())
        for _ in range(min(cfg.concurrency, len(batch) * cfg.group_size))
    ]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    if stopped is not None:
        raise stopped
    return sorted(results, key=lambda pair: pair[0].episode_id)


async def train_rl(cfg: TrainRLConfig, run: RunDir) -> Checkpoint:
    """Train all learners at a step barrier; publish the final handle only on success."""
    ctx, models, frozen_specs = _preflight(cfg)
    if cfg.tasks is None:
        raise ConfigError("train rl requires tasks")
    taskset = TaskSet.load(cfg.tasks)
    tasks = read_tasks(taskset)
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ConfigError("training TaskSet must be non-empty with unique task_id values")
    runlog.require_clean_tree()
    provenance = runlog.provenance()
    latest = _latest_checkpoint(run)
    completed = latest.step if latest else -1
    cursor = DataCursor(**latest.data_cursor) if latest else DataCursor()
    rae_state = dict(latest.rae_state) if latest else {}
    samplers: dict[str, SamplerSnapshot] = {}
    if latest:
        for name, record in latest.learners.items():
            if record["sampler"]:
                samplers[name] = SamplerSnapshot(
                    name,
                    record["version"],
                    latest.meta.get("sampler_paths", {}).get(name, record["sampler"]),
                    latest.policy_ref(name),
                )
    metrics = run.read_rows("metrics.jsonl")
    previous_spend = max((row["spend"]["spent_usd"] for row in metrics), default=0.0)
    if run.path("progress.json").exists():
        previous_spend = max(
            previous_spend, json.loads(run.path("progress.json").read_text())["spend"]["spent_usd"]
        )
    atomic_write_text(
        run.path("metrics.jsonl"),
        "".join(json.dumps(row) + "\n" for row in metrics if row["step"] <= completed),
    )
    # Remove only the known audit files of steps that must be repeated.
    for directory in run.path("rollouts").glob("step_*"):
        if int(directory.name[5:]) > completed:
            for filename in ("episodes.jsonl", "tokens.jsonl"):
                (directory / filename).unlink(missing_ok=True)
    spend = SpendGuard(cfg.max_usd)
    backends: dict[str, TrainBackend] = {}
    learners: dict[str, Learner] = {}
    run_name = cfg.run_name or run.config_hash[:12]
    inputs = (InputRef.of(taskset),)
    safe_to_save = True

    def progress() -> None:
        run.write_progress({"step": completed, "steps": cfg.steps, "spend": spend.summary()})

    async def save() -> Checkpoint:
        return await _save_checkpoint(
            run,
            learners,
            samplers,
            step=completed,
            run_name=run_name,
            cursor=cursor,
            rae_state=rae_state,
            previous=latest,
            inputs=inputs,
            meta={"provenance": provenance, "spend": spend.summary()},
        )

    try:
        spend.charge(previous_spend, "previous attempts")
        frozen, renderers = await build_policies(frozen_specs, spend=spend)
        for name, spec in cfg.learners.items():
            if spec.backend not in backends:
                kwargs = {"state_dir": run.path("states")} if spec.backend == "fake" else {}
                backends[spec.backend] = make_backend(
                    spec.backend, spend=spend, base_url=cfg.base_url, **kwargs
                )
            initial = (
                replace(
                    spec,
                    init_from=str(run.path(f"checkpoints/step_{latest.step:05d}/checkpoint.json")),
                )
                if latest
                else spec
            )
            learner = await backends[spec.backend].create_learner(
                name, initial, model=models[name], seed=derive_seed(cfg.seed, "learner", name)
            )
            learners[name] = learner
            if latest:
                await learner.load_state(latest.require_state(name), with_optimizer=True)
                learner.version = latest.learners[name]["version"]
            elif spec.init_from and not spec.init_from.startswith("tinker://"):
                warm = Checkpoint.load(spec.init_from)
                if warm.learners[name]["sampler"]:
                    samplers[name] = SamplerSnapshot(
                        name, learner.version, warm.learners[name]["sampler"], warm.policy_ref(name)
                    )
        for role, seat in cfg.seating.items():
            name = seat_learner(seat)
            if name is not None:
                model = models[name]
                renderers[role] = partial(get_renderer, model.renderer, hf_id=model.hf_id)
        for step in range(completed + 1, cfg.steps):
            batch, next_cursor = take_tasks(tasks, cursor, cfg.batch_tasks, seed=cfg.seed)
            policies = dict(frozen)
            for role, seat in cfg.seating.items():
                name = seat_learner(seat)
                if name is not None:
                    policies[role] = learners[name].policy(policy_id=role)
                    assert policies[role].trainable
            sampled = await _rollouts(
                cfg,
                run,
                batch,
                step=step,
                policies=policies,
                renderers=renderers,
                learners=learners,
                spend=spend,
            )
            episodes = [episode for episode, _ in sampled]
            credits, stats, next_rae = assign_credit(
                episodes, cfg.credit, replace(ctx, step=step), rae_state=rae_state
            )
            datums: dict[str, list[TrainDatum]] = {name: [] for name in learners}
            max_len = {name: model.max_ctx for name, model in models.items()}
            for episode, buffers in sampled:
                for datum in build_datums(episode, buffers, credits, max_len=max_len):
                    datums[datum.learner].append(datum)
            stepped = [name for name, data in datums.items() if data]
            # All Tinker reservations must fit before any optimizer is submitted.
            spend.check(
                sum(
                    tinker_cost(models[name], train=sum(len(d.tokens) - 1 for d in datums[name]))
                    for name in stepped
                    if cfg.learners[name].backend == "tinker"
                ),
                "all learners' training step",
            )
            safe_to_save = False
            results = await asyncio.gather(
                *(learners[name].train_step(datums[name]) for name in stepped),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            snapshots = await asyncio.gather(
                *(learners[name].sync_sampler(f"{run_name}-{name}-s{step}") for name in stepped),
                return_exceptions=True,
            )
            for snapshot in snapshots:
                if isinstance(snapshot, BaseException):
                    raise snapshot
                samplers[snapshot.learner] = snapshot
            completed, cursor, rae_state = step, next_cursor, next_rae
            safe_to_save = True
            run.append_row(
                "metrics.jsonl",
                {
                    "step": step,
                    "reward_mean": stats.reward_mean,
                    "accuracy": fmean(e.grades["_system"]["correct"] for e in episodes),
                    **{
                        key: fmean(e.metrics.get(key, 0.0) for e in episodes)
                        for key in ("total_gen", "calls", "n_workers", "cross_reads", "n_sessions")
                    },
                    "learners": {result.learner: asdict(result) for result in results},
                    "idle_learners": [name for name in learners if name not in stepped],
                    "credit": asdict(stats),
                    "datums": datum_stats([d for data in datums.values() for d in data]),
                    "spend": spend.summary(),
                },
            )
            progress()
            if (step + 1) % cfg.checkpoint_every == 0 or step == cfg.steps - 1:
                latest = await save()
        assert latest is not None
        return latest
    except BudgetExceededError:
        if (
            safe_to_save
            and len(learners) == len(cfg.learners)
            and (latest is None or latest.step != completed)
        ):
            latest = await save()
        raise
    finally:
        progress()
        await asyncio.gather(*(backend.close() for backend in backends.values()))
