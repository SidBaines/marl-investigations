"""Preserve exact sampled ids and raw logprobs across Tinker endpoints."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from math import isfinite
from typing import Any

from marli.budget import SpendGuard, tinker_cost
from marli.errors import BackendError, ConfigError
from marli.interact.types import Termination, Usage
from marli.model import ModelSpec
from marli.policy.base import CallMeta, Sample, SamplingSpec, check_trainable_sampling


class TinkerPolicy:
    """Bound calls to ``sample_async``; the SDK may retry transport operations internally."""

    def __init__(
        self,
        policy_id: str,
        sampling_client: Any,
        *,
        renderer_name: str,
        trainable: bool,
        policy_version: int | None = None,
        model: ModelSpec | None = None,
        spend: SpendGuard | None = None,
    ) -> None:
        self.policy_id = policy_id
        self.sampling_client = sampling_client
        self.renderer_name = renderer_name
        self.trainable = trainable
        self.policy_version = policy_version
        self.model = model
        self.spend = spend

    async def sample(
        self,
        prompt_ids: Sequence[int],
        spec: SamplingSpec,
        *,
        seed: int,
        meta: CallMeta | None = None,
    ) -> Sample:
        if self.trainable:
            check_trainable_sampling(spec, policy_id=self.policy_id)
        if not spec.stop_token_ids:
            raise ConfigError("Tinker sampling requires non-empty stop_token_ids")

        import tinker
        from tinker.lib.retry_handler import RetryConfig, is_retryable_status_code

        prompt = list(prompt_ids)
        if self.model is not None and self.spend is not None:
            self.spend.check(
                tinker_cost(self.model, prefill=len(prompt), sample=spec.max_tokens),
                self.policy_id,
            )
        model_input = tinker.ModelInput.from_ints(prompt)
        params = tinker.SamplingParams(
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
            top_p=spec.top_p,
            top_k=spec.top_k,
            stop=list(spec.stop_token_ids),
            seed=seed,
        )
        for attempt in range(3):
            try:
                response = await self.sampling_client.sample_async(
                    prompt=model_input, num_samples=1, sampling_params=params
                )
                break
            except Exception as exc:
                retryable = isinstance(exc, (*RetryConfig.retryable_exceptions, ConnectionError))
                if isinstance(exc, tinker.APIStatusError):
                    retryable = is_retryable_status_code(exc.status_code)
                if not retryable or attempt == 2:
                    raise BackendError(f"Tinker sampling failed: {exc}") from exc
                await asyncio.sleep(2.0**attempt)

        try:
            sequence = response.sequences[0]
            ids = tuple(sequence.tokens)
            logprobs = None if sequence.logprobs is None else tuple(sequence.logprobs)
            stop_reason = sequence.stop_reason
        except (AttributeError, IndexError, TypeError) as exc:
            raise BackendError("Tinker returned an invalid sampling response") from exc
        cost = 0.0
        if self.model is not None and self.spend is not None:
            cost = tinker_cost(self.model, prefill=len(prompt), sample=len(ids))
            self.spend.charge(cost, self.policy_id)
        if logprobs is None or len(logprobs) != len(ids):
            raise BackendError("Tinker completion token/logprob length mismatch")
        if any(not isinstance(token, int) or isinstance(token, bool) for token in ids):
            raise BackendError("Tinker completion ids must be integers (not bool)")
        if any(
            not isinstance(lp, (int, float)) or isinstance(lp, bool) or not isfinite(lp)
            for lp in logprobs
        ):
            raise BackendError("Tinker completion logprobs must be finite numbers")
        return Sample(
            completion_ids=ids,
            logprobs=logprobs,
            termination={"stop": Termination.STOP, "length": Termination.LENGTH}.get(
                stop_reason, Termination.ERROR
            ),
            policy_version=self.policy_version,
            usage=Usage(
                prompt_tokens=len(prompt),
                completion_tokens=len(ids),
                cached_prompt_tokens=getattr(response, "prompt_cache_hit_tokens", None),
                cost_usd=cost,
                tokenizer=self.renderer_name,
            ),
        )


def make_service_client(base_url: str | None) -> Any:
    """Reject implicit endpoint overrides before the SDK can route a paid call."""
    import tinker
    from tinker.lib.base_url import DEFAULT_BASE_URL

    explicit_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    env_url = (os.environ.get("TINKER_BASE_URL") or "").rstrip("/")
    if env_url and env_url != explicit_url:
        raise ConfigError(
            f"TINKER_BASE_URL {env_url!r} disagrees with policy base_url {explicit_url!r}"
        )
    return tinker.ServiceClient(base_url=explicit_url)


async def sampling_client_for(
    service: Any, *, base_model: str | None = None, model_path: str | None = None
) -> Any:
    """Let our three-attempt sample loop own sample-level retries.

    The service client's default transport retries remain enabled for session
    and sampling-session creation, which our sample loop does not cover.
    """
    from tinker.lib.retry_handler import RetryConfig

    try:
        return await service.create_sampling_client_async(
            base_model=base_model,
            model_path=model_path,
            retry_config=RetryConfig(enable_retry_logic=False),
        )
    except ValueError as exc:
        raise ConfigError(f"Invalid Tinker sampling client configuration: {exc}") from exc
    except Exception as exc:
        raise BackendError(f"Tinker sampling client creation failed: {exc}") from exc
