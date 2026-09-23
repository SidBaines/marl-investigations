"""Keep each LoRA's optimizer isolated while pipelining Tinker training requests.

Only submission is awaited before both operations are queued. Spend is reserved
before either request, and forward/backward is never retried by this layer.
"""

from __future__ import annotations

import asyncio
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
def _sdk_errors(operation: str) -> Iterator[None]:
    import tinker

    try:
        yield
    except MarliError:
        raise
    except (ValueError, tinker.BadRequestError, tinker.UnprocessableEntityError) as exc:
        raise ConfigError(f"Tinker {operation}: {exc}") from exc
    except Exception as exc:
        raise BackendError(f"Tinker {operation} failed: {exc}") from exc


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
        n_action_tokens = sum(d.n_action_tokens for d in datums)

        import tinker

        with _sdk_errors("datum/optimizer configuration"):
            data = [
                tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(list(d.tokens[:-1])),
                    loss_fn_inputs={
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
        with _sdk_errors("train_step"):
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
            diffs.extend(
                sample - train
                for sample, train, mask in zip(datum.logprobs, logprobs, datum.mask, strict=True)
                if mask > 0
            )
        if "loss:sum" not in forward_result.metrics:
            raise BackendError("Tinker forward/backward metrics missing loss:sum")
        optim_metrics = optim_result.metrics or {}
        return StepResult(
            learner=self.name,
            n_datums=len(datums),
            n_tokens=n_tokens,
            n_action_tokens=n_action_tokens,
            loss=forward_result.metrics["loss:sum"],
            grad_norm=optim_metrics.get("grad_norm"),
            kl_sample_train=sum(diffs) / len(diffs) if diffs else 0.0,
            metrics={**forward_result.metrics, **optim_metrics},
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
        with _sdk_errors("sync_sampler"):
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
        with _sdk_errors("save_state"):
            saved = await self.training_client.save_state_async(name, ttl_seconds=None)
            return (await saved.result_async()).path

    async def load_state(self, path: str, *, with_optimizer: bool = True) -> None:
        with _sdk_errors("load_state"):
            if with_optimizer:
                loaded = await self.training_client.load_state_with_optimizer_async(path)
            else:
                loaded = await self.training_client.load_state_async(path)
            await loaded.result_async()

    async def close(self) -> None:
        # TrainingClient has no public close; its backend owns the shared session.
        pass


class TinkerBackend:
    name = "tinker"

    def __init__(self, spend: SpendGuard | None, base_url: str | None = None) -> None:
        self.spend = spend
        self.base_url = base_url
        with _sdk_errors("service creation"):
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
        state = spec.init_from
        sampler = None
        if state is not None and not state.startswith("tinker://"):
            checkpoint = Checkpoint.load(state)
            state = checkpoint.require_state(name)
            if checkpoint.learners[name].get("sampler"):
                ref = parse_ref(checkpoint.policy_ref(name))
                if ref.kind != "tinker" or ref.sampler is None:
                    raise ConfigError("Tinker init_from requires a Tinker sampler checkpoint")
                sampler = ref.sampler
        with _sdk_errors("learner creation"):
            if state is None:
                client = await self.service.create_lora_training_client_async(
                    base_model=model.tinker_id, rank=spec.rank, seed=seed
                )
            else:
                client = await self.service.create_training_client_from_state_async(path=state)
                # SDK 0.30.1's constructor restores weights only, despite its name.
                restored = await client.load_state_with_optimizer_async(state)
                await restored.result_async()
            if sampler is None:
                sampling_client = await self.service.create_sampling_client_async(
                    base_model=model.tinker_id
                )
            else:
                sampling_client = await self.service.create_sampling_client_async(
                    model_path=sampler
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
        )
        self.learners[name] = learner
        return learner

    async def close(self) -> None:
        with _sdk_errors("close"):
            await self.service.close("success")
