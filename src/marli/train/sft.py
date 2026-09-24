"""Warm-start from exact teacher tokens with recoverable optimizer boundaries.

Cross-entropy is sum-reduced over completion positions. ``loss_agg=mean_per_token``
divides their weights by the batch's action-token count. Batches count model-input
tokens (len(tokens) - 1), never split segments, and never cross epoch boundaries.
Only saved checkpoints advance the recovery cursor; unsaved updates are replayed.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

from marli import runlog
from marli.budget import SpendGuard
from marli.config import doc_field, input_field, runtime_field
from marli.data.sft import SFTSet, read_datums
from marli.errors import ConfigError
from marli.handles import InputRef, atomic_write_text
from marli.model import load_model
from marli.policy.refs import parse_ref
from marli.render.registry import get_renderer
from marli.rundir import RunDir
from marli.seeds import derive_seed
from marli.train.backends.registry import make_backend
from marli.train.checkpoint import Checkpoint
from marli.train.loop import DataCursor, _latest_checkpoint, _save_checkpoint
from marli.train.types import LearnerSpec, TrainDatum


@dataclass
class TrainSFTConfig:
    data: str | None = input_field(None, help="SFTSet")
    learner: LearnerSpec = field(default_factory=lambda: LearnerSpec(loss="cross_entropy"))
    epochs: int = 1
    batch_tokens: int = 65536
    seed: int = 0
    checkpoint_every: int = 50
    run_name: str = ""
    loss_agg: str = doc_field("token_sum", help="token_sum | mean_per_token (batch action tokens)")
    max_usd: float | None = runtime_field(None)
    base_url: str | None = runtime_field(None)

    def __post_init__(self) -> None:
        self.learner.__post_init__()
        if self.learner.loss != "cross_entropy":
            raise ConfigError("train sft requires learner.loss=cross_entropy")
        for name in ("epochs", "batch_tokens", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ConfigError("seed must be an integer")
        if self.loss_agg not in {"token_sum", "mean_per_token"}:
            raise ConfigError("loss_agg must be token_sum or mean_per_token")
        if self.run_name and (
            self.run_name in {".", ".."}
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
                for c in self.run_name
            )
        ):
            raise ConfigError("run_name must contain only letters, digits, _.-")
        if self.base_url:
            url = urlsplit(self.base_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ConfigError("base_url must be an HTTP URL without credentials or query")
        SpendGuard(self.max_usd)


def batches(
    datums: Sequence[TrainDatum], cursor: DataCursor, *, epochs: int, batch_tokens: int, seed: int
) -> Iterator[tuple[list[TrainDatum], DataCursor]]:
    """Recreate seeded epoch permutations and greedy batches from a saved cursor."""
    for epoch in range(cursor.epoch, epochs):
        order = list(datums)
        random.Random(derive_seed(seed, "sft", epoch)).shuffle(order)
        index = cursor.index if epoch == cursor.epoch else 0
        while index < len(order):
            batch: list[TrainDatum] = []
            n_tokens = 0
            while index < len(order):
                datum = order[index]
                length = len(datum.tokens) - 1
                if length > batch_tokens:
                    raise ConfigError("datum exceeds batch_tokens; reduce data sft max_len")
                if n_tokens + length > batch_tokens:
                    break
                batch.append(datum)
                n_tokens += length
                index += 1
            next_cursor = (
                DataCursor(epoch + 1, 0) if index == len(order) else DataCursor(epoch, index)
            )
            yield batch, next_cursor


async def train_sft(cfg: TrainSFTConfig, run: RunDir) -> Checkpoint:
    """Train the ``student`` learner; its checkpoint name is also the RL init_from seat.

    FakeBackend retains a scripted teacher factory for its eval stand-in, as fake
    weights cannot generate tokens. Tinker checkpoints publish the trained sampler.
    """
    cfg.__post_init__()
    if not cfg.data or not cfg.learner.base_model:
        raise ConfigError("train sft requires data and learner.base_model")
    data = SFTSet.load(cfg.data)
    model = load_model(cfg.learner.base_model)
    if get_renderer(model.renderer, hf_id=model.hf_id).tokenizer_sha != data.tokenizer_sha:
        raise ConfigError("SFTSet tokenizer_sha differs from learner tokenizer")
    datums = list(read_datums(data))
    if len(datums) != data.n:
        raise ConfigError("SFTSet datum count differs from its manifest")
    if not datums:
        raise ConfigError("training SFTSet must not be empty")
    if cfg.learner.backend == "local":
        raise ConfigError("local training backend not implemented until M4")
    max_len = model.max_ctx
    if cfg.learner.backend == "tinker":
        if model.tinker_max_ctx is None:
            raise ConfigError("student model has no tinker_max_ctx")
        max_len = min(max_len, model.tinker_max_ctx)
    for datum in datums:
        if len(datum.tokens) > max_len:
            raise ConfigError(f"datum exceeds learner max ctx {max_len}")
        if len(datum.tokens) - 1 > cfg.batch_tokens:
            raise ConfigError("datum exceeds batch_tokens; reduce data sft max_len")
        if not any(datum.mask) or any(weight not in {0.0, 1.0} for weight in datum.mask):
            raise ConfigError("SFTSet masks must be binary with at least one action token")
    kwargs: dict[str, Any] = {}
    if cfg.learner.backend == "fake":
        factories = {
            parse_ref(ref).target
            for ref in data.teacher_policies.values()
            if parse_ref(ref).kind == "scripted"
        }
        if len(factories) != 1:
            raise ConfigError("fake SFT requires one scripted teacher factory for its sampler")
        kwargs = {"state_dir": run.path("states"), "policy_factory": factories.pop()}
    runlog.require_clean_tree()
    provenance = runlog.provenance()
    latest = _latest_checkpoint(run)
    completed = latest.step if latest else -1
    cursor = DataCursor(**latest.data_cursor) if latest else DataCursor()
    metrics = run.read_rows("metrics.jsonl")
    previous_spend = max((row["spend"]["spent_usd"] for row in metrics), default=0.0)
    if latest:
        previous_spend = max(previous_spend, latest.meta["spend"]["spent_usd"])
    if run.path("progress.json").exists():
        previous_spend = max(
            previous_spend, json.loads(run.path("progress.json").read_text())["spend"]["spent_usd"]
        )
    atomic_write_text(
        run.path("metrics.jsonl"),
        "".join(json.dumps(row) + "\n" for row in metrics if row["step"] <= completed),
    )
    spend = SpendGuard(cfg.max_usd)
    spend.charge(previous_spend, "previous attempts")
    initial = (
        replace(
            cfg.learner,
            init_from=str(run.path(f"checkpoints/step_{latest.step:05d}/checkpoint.json")),
        )
        if latest
        else cfg.learner
    )
    inputs = (InputRef.of(data),)
    if cfg.learner.init_from and not cfg.learner.init_from.startswith("tinker://"):
        inputs += (InputRef.of(Checkpoint.load(cfg.learner.init_from)),)
    run_name = cfg.run_name or run.config_hash[:12]
    backend = make_backend(cfg.learner.backend, spend=spend, base_url=cfg.base_url, **kwargs)
    try:
        learner = await backend.create_learner(
            "student", initial, model=model, seed=derive_seed(cfg.seed, "learner", "student")
        )
        for step, (batch, next_cursor) in enumerate(
            batches(
                datums, cursor, epochs=cfg.epochs, batch_tokens=cfg.batch_tokens, seed=cfg.seed
            ),
            start=completed + 1,
        ):
            n_action_tokens = sum(d.n_action_tokens for d in batch)
            scale = 1.0 / n_action_tokens if cfg.loss_agg == "mean_per_token" else 1.0
            # Learners guard versions for every loss. SFT does not use behavior
            # logprobs; only submitted copies adopt the student's sampler version.
            submitted = [
                replace(
                    d,
                    learner="student",
                    policy_version=learner.version,
                    mask=tuple(weight * scale for weight in d.mask),
                )
                for d in batch
            ]
            started = perf_counter()
            result = await learner.train_step(submitted)
            elapsed = perf_counter() - started
            completed, cursor = step, next_cursor
            run.append_row(
                "metrics.jsonl",
                {
                    "step": step,
                    "loss": result.loss,
                    "n_tokens": result.n_tokens,
                    "n_action_tokens": n_action_tokens,
                    "tokens/sec": result.n_tokens / elapsed,
                    "spend": spend.summary(),
                },
            )
            run.write_progress({"step": completed, "spend": spend.summary()})
            if (step + 1) % cfg.checkpoint_every == 0 or cursor.epoch == cfg.epochs:
                snapshot = await learner.sync_sampler(f"{run_name}-student-s{step}")
                latest = await _save_checkpoint(
                    run,
                    {"student": learner},
                    {"student": snapshot},
                    step=step,
                    run_name=run_name,
                    cursor=cursor,
                    rae_state={},
                    previous=latest,
                    inputs=inputs,
                    meta={"provenance": provenance, "spend": spend.summary()},
                )
        assert latest is not None
        return latest
    finally:
        run.write_progress({"step": completed, "spend": spend.summary()})
        await backend.close()
