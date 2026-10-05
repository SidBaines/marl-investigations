"""Resolve a cell's policy to an OpenAI-compatible endpoint and record what it serves.

Policies use the usual vLLM refs (``vllm:@<server.json>#<model>`` or
``vllm:<url>#<model>``): ``<model>`` is the served name, the base model or a LoRA
adapter loaded under that name. Before any sampling the verb asks the server's
``/v1/models`` for the name (a missing adapter is an error, not an empty result) and
records vLLM's ``root`` (the adapter path) and ``parent``; when the root is a local
adapter directory, its files' sha256 go into the manifest.

``inspect:mockllm/<name>`` refs select Inspect's offline mock model (tests only).
Paid providers are deliberately not reachable from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from marli.errors import BackendError, ConfigError
from marli.eval.external.spec import GENERATION_KEYS, ExternalSuite, check_generation
from marli.handles import sha256_file
from marli.policy.refs import parse_ref

MOCK_PREFIX = "inspect:mockllm/"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")


@dataclass(frozen=True)
class Endpoint:
    ref: str
    provider: str  # vllm | mockllm
    model: str  # served model name
    base_url: str | None = None  # server root, without /v1
    server_json: str | None = None
    hf_id: str | None = None

    @property
    def api_base(self) -> str:
        assert self.base_url is not None
        return self.base_url + "/v1"


def resolve_endpoint(ref: str) -> Endpoint:
    if ref.startswith(MOCK_PREFIX):
        return Endpoint(ref, "mockllm", ref.removeprefix(MOCK_PREFIX))
    parsed = parse_ref(ref, resolve_paths=True)
    if parsed.kind != "vllm":
        raise ConfigError(
            f"external evals take vllm refs (or {MOCK_PREFIX}<name> in tests), got {ref!r}"
        )
    hf_id = None
    base_url = parsed.base_url
    if parsed.server_json is not None:
        from marli.policy.vllm import read_server_json

        base_url, models = read_server_json(parsed.server_json)
        if parsed.target not in models:
            raise ConfigError(f"model {parsed.target!r} is not listed in {parsed.server_json}")
        hf_id = json.loads(Path(parsed.server_json).read_text(encoding="utf-8")).get("hf_id")
    assert base_url is not None
    base_url = base_url.rstrip("/").removesuffix("/v1")
    return Endpoint(ref, "vllm", parsed.target, base_url, parsed.server_json, hf_id)


async def probe(endpoint: Endpoint, *, timeout_s: float) -> dict[str, Any]:
    """The server's own record of the served name; fails loudly when it is absent."""
    if endpoint.provider == "mockllm":
        return {"model": endpoint.model, "provider": "mockllm"}
    async with httpx.AsyncClient(trust_env=False, timeout=timeout_s) as client:
        try:
            response = await client.get(endpoint.api_base + "/models")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"{endpoint.api_base}/models unreachable: {exc}") from exc
    entries = {entry.get("id"): entry for entry in response.json().get("data", [])}
    if endpoint.model not in entries:
        raise BackendError(
            f"{endpoint.model!r} is not served at {endpoint.api_base} (serving {sorted(entries)})"
        )
    entry = entries[endpoint.model]
    record: dict[str, Any] = {
        "model": endpoint.model,
        "base_url": endpoint.base_url,
        "server_json": endpoint.server_json,
        "hf_id": endpoint.hf_id,
        "root": entry.get("root"),
        "parent": entry.get("parent"),
        "adapter_sha256": None,
    }
    root = entry.get("root")
    if entry.get("parent") and isinstance(root, str) and Path(root).is_dir():
        record["adapter_sha256"] = {
            name: sha256_file(Path(root) / name)
            for name in ADAPTER_FILES
            if (Path(root) / name).is_file()
        }
    return record


def resolve_settings(
    suite: ExternalSuite, model_generation: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Model-card defaults < the benchmark's own settings < explicit overrides.

    ``chat_template_kwargs`` carry the thinking switch: ``enable_thinking`` is set from
    the suite and may not be contradicted. Other template kwargs (e.g. Qwen3.8's
    ``reasoning_effort``) come from the model settings so they match training.
    """
    check_generation(model_generation, "model_generation")
    check_generation(overrides, "generation_overrides")
    settings = {**model_generation, **suite.generation, **overrides}
    kwargs = dict(model_generation.get("chat_template_kwargs") or {})
    kwargs.update(overrides.get("chat_template_kwargs") or {})
    thinking = suite.thinking == "on"
    if kwargs.get("enable_thinking", thinking) != thinking:
        raise ConfigError(
            f"suite {suite.name!r} fixes thinking {suite.thinking!r}; "
            "chat_template_kwargs.enable_thinking contradicts it"
        )
    kwargs["enable_thinking"] = thinking
    settings["chat_template_kwargs"] = kwargs
    return {key: settings[key] for key in GENERATION_KEYS if settings.get(key) is not None}


def request_fields(settings: dict[str, Any]) -> dict[str, Any]:
    """Settings as OpenAI chat-completions request fields (vLLM extensions in extra_body)."""
    standard = (
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    )
    request = {key: settings[key] for key in standard if key in settings}
    extra = {key: settings[key] for key in settings if key not in standard}
    if extra:
        request["extra_body"] = extra
    return request
