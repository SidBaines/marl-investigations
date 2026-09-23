"""Keep vLLM completions token-native so training never reconstructs server text."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import httpx

from marli.errors import BackendError, ConfigError
from marli.interact.types import Termination, Usage
from marli.policy.base import CallMeta, Sample, SamplingSpec, check_trainable_sampling


class VLLMPolicy:
    def __init__(
        self,
        policy_id: str,
        base_url: str,
        model: str,
        *,
        renderer_name: str,
        trainable: bool,
        policy_version: int | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 600,
    ) -> None:
        self.policy_id = policy_id
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.renderer_name = renderer_name
        self.trainable = trainable
        self.policy_version = policy_version
        self.timeout_s = timeout_s
        self._owns_client = client is None
        self.client = client if client is not None else httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

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
            raise ConfigError("vLLM sampling requires non-empty stop_token_ids")
        prompt = list(prompt_ids)
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "top_k": spec.top_k,
            "seed": seed,
            "logprobs": 1,
            "return_token_ids": True,
            "stop_token_ids": list(spec.stop_token_ids),
            "skip_special_tokens": False,
            "include_stop_str_in_output": True,
        }
        root = self.base_url if self.base_url.endswith("/v1") else self.base_url + "/v1"
        for attempt in range(3):
            try:
                response = await self.client.post(
                    root + "/completions", json=body, timeout=self.timeout_s
                )
                response.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    message = f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"
                    retryable = 500 <= exc.response.status_code < 600
                else:
                    message = str(exc)
                    retryable = isinstance(
                        exc,
                        (
                            httpx.ConnectError,
                            httpx.RemoteProtocolError,
                            httpx.ReadError,
                            httpx.WriteError,
                        ),
                    )
                if not retryable or attempt == 2:
                    raise BackendError(f"vLLM sampling failed: {message}") from exc
                await asyncio.sleep(2.0**attempt)

        try:
            choice = response.json()["choices"][0]
            ids = tuple(choice["token_ids"])
            logprobs = tuple(choice["logprobs"]["token_logprobs"])
            finish_reason = choice["finish_reason"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError("vLLM returned an invalid token sampling response") from exc
        if len(logprobs) != len(ids):
            raise BackendError("vLLM completion token/logprob length mismatch")
        if any(not isinstance(token, int) or isinstance(token, bool) for token in ids):
            raise BackendError("vLLM completion ids must be integers (not bool)")
        if any(
            not isinstance(lp, (int, float)) or isinstance(lp, bool) or not isfinite(lp)
            for lp in logprobs
        ):
            raise BackendError("vLLM completion logprobs must be finite numbers")
        return Sample(
            completion_ids=ids,
            logprobs=logprobs,
            termination={"stop": Termination.STOP, "length": Termination.LENGTH}.get(
                finish_reason, Termination.ERROR
            ),
            policy_version=self.policy_version,
            usage=Usage(
                prompt_tokens=len(prompt),
                completion_tokens=len(ids),
                tokenizer=self.renderer_name,
            ),
        )


def read_server_json(path: str | Path) -> tuple[str, dict[str, str]]:
    """Read the endpoint and index the manifest's served model names by name."""
    try:
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        base_url = data["base_url"]
        models = data["models"]
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(models, list) or any(not isinstance(m, str) or not m for m in models):
            raise ValueError("models must be a list of non-empty model names")
        return base_url, {name: name for name in models}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"Cannot read vLLM server manifest {path}: {exc}") from exc
