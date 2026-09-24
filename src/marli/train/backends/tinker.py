"""Keep each LoRA's optimizer isolated while pipelining Tinker training requests.

Only submission is awaited before both operations are queued. Spend is reserved
before either request, and forward/backward is never retried by this layer.

``init_from`` resumes weights and optimizer by default. The explicit backend
kwarg ``init_mode="weights"`` warm-starts with a fresh optimizer instead.
Checkpoint manifests restore their sampler version; raw state paths start at
version zero. Both modes publish restored weights before serving a policy when
the record has no same-step sampler. ``load_state`` follows the same rules and
keeps the current version for raw paths, without incrementing it like a sync.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from marli.budget import SpendGuard, tinker_cost
from marli.errors import BackendError, ConfigError, MarliError
from marli.model import ModelSpec
from marli.policy.refs import PolicyRef, parse_ref
from marli.policy.tinker import TinkerPolicy, make_service_client
from marli.train.backends.base import SamplerSnapshot, StepResult
from marli.train.checkpoint import Checkpoint
from marli.train.types import LearnerSpec, TrainDatum


@contextmanager
def _sdk_errors(
    operation: str, *, configuration: bool = False, failures: list[BaseException] | None = None
) -> Iterator[None]:
    try:
        try:
            import tinker
        except ImportError as exc:
            raise ConfigError("Tinker backend requires the [tinker] extra") from exc
        try:
            yield
        except MarliError:
            raise
        except (tinker.BadRequestError, tinker.UnprocessableEntityError) as exc:
            raise ConfigError(f"Tinker {operation}: {exc}") from exc
        except Exception as exc:
            error = ConfigError if configuration and isinstance(exc, ValueError) else BackendError
            raise error(f"Tinker {operation} failed: {exc}") from exc
    except BaseException as exc:
        if failures is not None:
            failures.append(exc)
        raise


class TinkerLearner:
    def __init__(
        self,
        name: str,
        spec: LearnerSpec,
        *,
        model: ModelSpec,
        service: Any,
        training_client: Any,
        sampling_client: Any,
        spend: SpendGuard | None,
        base_url: str | None,
        failures: list[BaseException] | None = None,
    ) -> None:
        self.name = name
        self.spec = spec
        self.model = model
        self.version = 0
        self.service = service
        self.training_client = training_client
        self.sampling_client = sampling_client
        self.spend = spend
        self.base_url = base_url
        self._failures = failures if failures is not None else []
        self._sampler_names: set[str] = set()

    async def train_step(
        self, datums: Sequence[TrainDatum], *, learning_rate: float | None = None
    ) -> StepResult:
        if not datums:
            raise ConfigError("train_step requires non-empty datums")
        for datum in datums:
            if datum.learner != self.name:
                raise ConfigError(f"datum for {datum.learner!r} routed to learner {self.name!r}")
            if len(datum.tokens) > self.model.max_ctx:
                raise ConfigError(f"datum exceeds model max ctx {self.model.max_ctx}")
        versions = {datum.policy_version for datum in datums}
        if len(versions) != 1 or versions != {self.version}:
            raise ConfigError(f"datums must use current policy version {self.version}")
        n_tokens = sum(len(d.tokens) - 1 for d in datums)
        cross_entropy = self.spec.loss == "cross_entropy"
        n_action_tokens = (
            sum(m > 0 for d in datums for m in d.mask)
            if cross_entropy
            else sum(d.n_action_tokens for d in datums)
        )

        with _sdk_errors(
            "datum/optimizer configuration", configuration=True, failures=self._failures
        ):
            import tinker

            data = [
                tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(list(d.tokens[:-1])),
                    # SDK 0.30.1 coerces target lists to int64, weights to float32.
                    loss_fn_inputs={"target_tokens": list(d.tokens[1:]), "weights": list(d.mask)}
                    if cross_entropy
                    else {
                        "target_tokens": list(d.tokens[1:]),
                        "logprobs": list(d.logprobs),
                        "advantages": list(d.advantages),
                    },
                )
                for d in datums
            ]
            adam = tinker.AdamParams(
                learning_rate=self.spec.learning_rate if learning_rate is None else learning_rate,
                beta1=self.spec.beta1,
                beta2=self.spec.beta2,
                eps=self.spec.eps,
                weight_decay=self.spec.weight_decay,
                grad_clip_norm=self.spec.grad_clip,
            )
        if self.spend is not None:
            if self.model.tinker_prices is not None:
                cost = tinker_cost(self.model, train=n_tokens)
                # check() avoids recording spend for a request we will never submit.
                self.spend.check(cost, self.name)
                self.spend.charge(cost, self.name)
            elif self.spend.remaining() is not None:
                raise ConfigError(f"model {self.model.name!r} has no Tinker prices")
        with _sdk_errors("train_step", failures=self._failures):
            forward = await self.training_client.forward_backward_async(
                data, loss_fn=self.spec.loss
            )
            optim = await self.training_client.optim_step_async(adam)
            forward_result, optim_result = await asyncio.gather(
                forward.result_async(), optim.result_async()
            )

            diffs: list[float] = []
            outputs = forward_result.loss_fn_outputs
            if len(outputs) != len(datums):
                raise BackendError("Tinker training output datum count mismatch")
            for datum, output in zip(datums, outputs, strict=True):
                logprobs = output["logprobs"].tolist()
                if len(logprobs) != len(datum.logprobs):
                    raise BackendError("Tinker training logprob length mismatch")
                if cross_entropy:
                    continue
                diffs.extend(
                    sample - train
                    for sample, train, mask in zip(
                        datum.logprobs, logprobs, datum.mask, strict=True
                    )
                    if mask > 0
                )
            if "loss:sum" not in forward_result.metrics:
                raise BackendError("Tinker forward/backward metrics missing loss:sum")
            optim_metrics = optim_result.metrics or {}
            metrics = {**forward_result.metrics, **optim_metrics}
            return StepResult(
                learner=self.name,
                n_datums=len(datums),
                n_tokens=n_tokens,
                n_action_tokens=n_action_tokens,
                loss=forward_result.metrics["loss:sum"],
                grad_norm=metrics.get("grad_norm"),
                kl_sample_train=None
                if cross_entropy
                else (sum(diffs) / len(diffs) if diffs else 0.0),
                metrics=metrics,
            )

    def policy(self, *, policy_id: str | None = None) -> TinkerPolicy:
        return TinkerPolicy(
            self.name if policy_id is None else policy_id,
            self.sampling_client,
            trainable=True,
            policy_version=self.version,
            renderer_name=self.model.renderer,
            model=self.model,
            spend=self.spend,
        )

    async def sync_sampler(self, name: str) -> SamplerSnapshot:
        if name in self._sampler_names:
            raise ValueError(f"sampler name {name!r} already used by learner {self.name!r}")
        # A failed response may still have saved the remote name; never overwrite it.
        self._sampler_names.add(name)
        with _sdk_errors("sync_sampler", failures=self._failures):
            saved = await self.training_client.save_weights_for_sampler_async(
                name, ttl_seconds=None
            )
            path = (await saved.result_async()).path
            sampling_client = await self.service.create_sampling_client_async(model_path=path)
        ref = str(PolicyRef("tinker", self.model.tinker_id, base_url=self.base_url, sampler=path))
        self.sampling_client = sampling_client
        self.version += 1
        return SamplerSnapshot(self.name, self.version, path, ref)

    async def save_state(self, name: str) -> str:
        with _sdk_errors("save_state", failures=self._failures):
            saved = await self.training_client.save_state_async(name, ttl_seconds=None)
            return (await saved.result_async()).path

    async def load_state(self, path: str, *, with_optimizer: bool = True) -> None:
        sampler = None
        version = self.version
        if not path.startswith("tinker://"):
            checkpoint = Checkpoint.load(path)
            path = checkpoint.require_state(self.name)
            record = checkpoint.learners[self.name]
            if record["backend"] != "tinker":
                raise ConfigError("Tinker init_from requires backend=tinker")
            if record["base_model"] not in {self.model.name, self.model.tinker_id}:
                raise ConfigError("Tinker checkpoint base_model does not match the model")
            if record["rank"] != self.spec.rank:
                raise ConfigError("Tinker checkpoint rank does not match learner.rank")
            version = record["version"]
            if record.get("sampler"):
                ref = parse_ref(checkpoint.policy_ref(self.name))
                if ref.kind != "tinker" or ref.sampler is None:
                    raise ConfigError("Tinker init_from requires a Tinker sampler checkpoint")
                sampler = ref.sampler
        if not path.startswith("tinker://") or "sampler_weights" in path.split("/"):
            raise ConfigError("Tinker load_state requires a Tinker state path, not sampler weights")
        if self.spec.base_model and self.spec.base_model != self.model.name:
            raise ConfigError("learner.base_model does not match the supplied model")
        with _sdk_errors("load_state", failures=self._failures):
            info = await self.service.create_rest_client().get_weights_info_by_tinker_path(path)
            if info.base_model != self.model.tinker_id:
                raise ConfigError("Tinker state base_model does not match the model")
            if not info.is_lora or info.lora_rank != self.spec.rank:
                raise ConfigError("Tinker state rank does not match learner.rank")
            if with_optimizer:
                client = await self.service.create_training_client_from_state_with_optimizer_async(
                    path=path
                )
            else:
                client = await self.service.create_training_client_from_state_async(path=path)
            if sampler is None:
                name = f"{self.name}-init"
                suffix = 1
                while name in self._sampler_names:
                    name = f"{self.name}-restore-{suffix}"
                    suffix += 1
                self._sampler_names.add(name)
                saved = await client.save_weights_for_sampler_async(name, ttl_seconds=None)
                sampler = (await saved.result_async()).path
            sampling_client = await self.service.create_sampling_client_async(model_path=sampler)
        self.training_client = client
        self.sampling_client = sampling_client
        self.version = version

    async def close(self) -> None:
        # TrainingClient has no public close; its backend owns the shared session.
        pass


class TinkerBackend:
    name = "tinker"

    def __init__(
        self, spend: SpendGuard | None, base_url: str | None = None, *, init_mode: str = "resume"
    ) -> None:
        if init_mode not in {"resume", "weights"}:
            raise ConfigError("Tinker init_mode must be resume or weights")
        self.spend = spend
        self.base_url = base_url
        self.init_mode = init_mode
        self._failures: list[BaseException] = []
        with _sdk_errors("service creation", configuration=True, failures=self._failures):
            self.service = make_service_client(base_url)
        self.learners: dict[str, TinkerLearner] = {}

    async def create_learner(
        self, name: str, spec: LearnerSpec, *, model: ModelSpec, seed: int
    ) -> TinkerLearner:
        if name in self.learners:
            raise ConfigError(f"learner {name!r} already exists")
        if model.tinker_id is None:
            raise ConfigError(f"model {model.name!r} has no tinker_id")
        if (
            model.tinker_prices is None
            and self.spend is not None
            and self.spend.remaining() is not None
        ):
            raise ConfigError(f"model {model.name!r} has no Tinker prices")
        client = sampling_client = None
        if spec.init_from is None:
            with _sdk_errors("learner creation", failures=self._failures):
                client = await self.service.create_lora_training_client_async(
                    base_model=model.tinker_id, rank=spec.rank, seed=seed
                )
                sampling_client = await self.service.create_sampling_client_async(
                    base_model=model.tinker_id
                )
        learner = TinkerLearner(
            name,
            spec,
            model=model,
            service=self.service,
            training_client=client,
            sampling_client=sampling_client,
            spend=self.spend,
            base_url=self.base_url,
            failures=self._failures,
        )
        if spec.init_from is not None:
            await learner.load_state(spec.init_from, with_optimizer=self.init_mode == "resume")
        self.learners[name] = learner
        return learner

    async def close(self) -> None:
        error = sys.exception() or (self._failures[0] if self._failures else None)
        status = "success"
        if error is not None:
            status = (
                "interrupted"
                if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt))
                else "errored"
            )
        with _sdk_errors("close"):
            await self.service.close(status)
