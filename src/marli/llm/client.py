"""Keep provider transport differences out of callers' OpenAI-shaped requests.

Ported from scimt (ArcadiaImpact/science-of-midtraining @ 3cb3541)
src/scimt/utils/client.py, itself vendored from aligne v0.6.0.

Caching is opt-in: independent samples must never silently collapse into a
replayed response. Enabled caches require an explicit per-call salt and key
the canonical payload before wire translation. Credentials stay in headers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

RETRYABLE_STATUS = {408, 409, 429, *range(500, 600)}
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
_PROVIDERS = ("openai", "anthropic")
_ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
_ANTHROPIC_PASSTHROUGH = {
    "model",
    "top_p",
    "stop_sequences",
    "metadata",
    "thinking",
    "output_config",
}


class UnsupportedRequestError(RuntimeError):
    """A request the provider cannot support; retrying it unchanged cannot help."""


@dataclass(frozen=True)
class Endpoint:
    """A model and transport, with credentials resolved only when requesting.

    OpenAI-compatible base URLs include ``/v1``; Anthropic uses the bare host.
    An unset ``api_key_env`` permits unauthenticated local endpoints.
    ``extra_params`` are payload defaults and participate in the cache key.
    """

    base_url: str
    model: str
    api_key_env: str | None = None
    provider: str = "openai"
    extra_params: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.provider not in _PROVIDERS:
            raise ValueError(f"provider must be one of {_PROVIDERS}, got {self.provider!r}")
        if self.extra_params and "model" in self.extra_params:
            raise ValueError(
                "extra_params must not override reserved key 'model'; set Endpoint.model"
            )

    def headers(self) -> dict[str, str]:
        key = None
        if self.api_key_env is not None:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ValueError(f"missing API key environment variable {self.api_key_env!r}")
        if self.provider == "anthropic":
            headers = {"anthropic-version": ANTHROPIC_VERSION}
            if key:
                headers["x-api-key"] = key
            return headers
        return {"Authorization": f"Bearer {key}"} if key else {}


def _anthropic_tool_choice(choice: Any) -> dict[str, str]:
    if isinstance(choice, str) and choice in ("auto", "required", "none"):
        return {"type": "any" if choice == "required" else choice}
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "tool", "name": choice["function"]["name"]}
    raise UnsupportedRequestError("unsupported anthropic tool_choice")


def to_anthropic(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate text and function tools to Messages, rejecting unknown keys.

    System turns are joined; consecutive tool results share one user turn.
    Default temperature is omitted because recent Claude models reject it.
    """
    body: dict[str, Any] = {}
    for key, value in payload.items():
        if key in ("messages", "temperature", "max_tokens"):
            continue
        if key == "stop":
            body["stop_sequences"] = [value] if isinstance(value, str) else list(value)
        elif key == "tools":
            tools = []
            for tool in value:
                if tool.get("type") != "function":
                    raise UnsupportedRequestError("anthropic tools must be OpenAI function tools")
                function = tool["function"]
                translated = {
                    "name": function["name"],
                    "input_schema": function["parameters"],
                }
                if "description" in function:
                    translated["description"] = function["description"]
                tools.append(translated)
            body["tools"] = tools
        elif key == "tool_choice":
            body[key] = _anthropic_tool_choice(value)
        elif key in _ANTHROPIC_PASSTHROUGH:
            body[key] = value
        else:
            raise UnsupportedRequestError(
                f"payload key {key!r} is not supported on the anthropic provider"
            )

    system: list[str] = []
    messages: list[dict[str, Any]] = []
    previous_role = None
    for message in payload["messages"]:
        role, content = message["role"], message.get("content")
        calls = message.get("tool_calls") if role == "assistant" else None
        if content is None and calls:
            content = ""
        if not isinstance(content, str):
            raise UnsupportedRequestError(
                "the anthropic provider supports plain-string message content only, "
                f"got {type(content).__name__}"
            )
        if role == "system":
            system.append(content)
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": content,
            }
            if previous_role == "tool":
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
        elif role in ("user", "assistant"):
            if calls:
                blocks = [{"type": "text", "text": content}] if content else []
                for call in calls:
                    if call.get("type") != "function":
                        raise UnsupportedRequestError("anthropic tool calls must be functions")
                    function = call["function"]
                    try:
                        arguments = json.loads(function["arguments"])
                    except (ValueError, TypeError) as exc:
                        raise UnsupportedRequestError(
                            "tool arguments must be a JSON object"
                        ) from exc
                    if not isinstance(arguments, dict):
                        raise UnsupportedRequestError("tool arguments must be a JSON object")
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": function["name"],
                            "input": arguments,
                        }
                    )
                messages.append({"role": role, "content": blocks})
            else:
                messages.append({"role": role, "content": content})
        else:
            raise UnsupportedRequestError(f"message role {role!r} is not supported on anthropic")
        previous_role = role

    body["messages"] = messages
    body["max_tokens"] = payload.get("max_tokens", _ANTHROPIC_DEFAULT_MAX_TOKENS)
    if system:
        body["system"] = "\n\n".join(system)
    temperature = payload.get("temperature")
    if temperature is not None and temperature != 1.0:
        body["temperature"] = temperature
    return body


def from_anthropic(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize text, tool use and thinking without discarding provider usage."""
    if data.get("type") == "error" or data.get("error"):
        raise UnsupportedRequestError("anthropic error response")
    blocks = data.get("content", [])
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
    }
    calls = [
        {
            "id": block["id"],
            "type": "function",
            "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
        }
        for block in blocks
        if block.get("type") == "tool_use"
    ]
    if calls:
        message["tool_calls"] = calls
    thinking = [b.get("thinking", "") for b in blocks if b.get("type") == "thinking"]
    if thinking:
        message["reasoning_content"] = "".join(thinking)
    stop = data.get("stop_reason")
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "choices": [
            {
                "message": message,
                "finish_reason": "tool_calls" if stop == "tool_use" else stop,
            }
        ],
        "usage": data.get("usage", {}),
    }


def usage_of(response: dict[str, Any]) -> dict[str, int | float | None]:
    """Extract comparable counts and reported cost; unavailable cache/cost is None.

    Anthropic's input count excludes cache creation and reads, so those are
    added back to obtain the full prompt count. The response itself is unchanged.
    """
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    prompt = usage.get("prompt_tokens")
    if prompt is None:
        prompt = sum(
            usage.get(key, 0) or 0
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        "cached_prompt_tokens": details.get("cached_tokens", usage.get("cache_read_input_tokens")),
        "cost_usd": usage.get("cost"),
    }


def _load_cache_records(path: Path) -> list[dict[str, Any]]:
    """Repair only an interrupted final append; interior corruption stays loud."""
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    offset = 0
    for index, raw in enumerate(lines):
        if raw.strip():
            try:
                records.append(json.loads(raw))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if index != len(lines) - 1 or raw.endswith(b"\n"):
                    raise ValueError(f"malformed cache record {index + 1} in {path}") from exc
                with path.open("r+b") as cache_file:
                    cache_file.truncate(offset)
                    cache_file.flush()
                    os.fsync(cache_file.fileno())
                warnings.warn(f"truncated trailing cache fragment in {path}", stacklevel=2)
                return records
        offset += len(raw)
    # A complete JSON record can survive a crash before its newline does.
    if data and not data.endswith(b"\n"):
        with path.open("ab") as cache_file:
            cache_file.write(b"\n")
            cache_file.flush()
            os.fsync(cache_file.fileno())
    return records


@dataclass
class ChatClient:
    """Raw httpx transport with bounded retries and explicitly salted caching."""

    endpoint: Endpoint
    concurrency: int = 32
    max_retries: int = 6
    timeout: float = 120.0
    cache: str = "off"
    cache_path: Path | None = None
    request_semaphore: asyncio.Semaphore | None = field(default=None, repr=False)

    _sem: asyncio.Semaphore = field(init=False, repr=False)
    _cache: dict[str, dict[str, Any]] = field(init=False, repr=False)
    _cache_lock: asyncio.Lock = field(init=False, repr=False)
    _http: httpx.AsyncClient = field(init=False, repr=False)
    _use_max_completion_tokens: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError(f"concurrency must be > 0, got {self.concurrency}")
        if self.max_retries <= 0:
            raise ValueError(f"max_retries must be > 0, got {self.max_retries}")
        if self.cache not in ("off", "memory", "disk"):
            raise ValueError("cache must be one of 'off', 'memory', 'disk'")
        if self.cache == "disk" and self.cache_path is None:
            raise ValueError("disk cache requires cache_path")
        self._sem = self.request_semaphore or asyncio.Semaphore(self.concurrency)
        self._cache_lock = asyncio.Lock()
        self._cache = {}
        if self.cache == "disk" and self.cache_path is not None and self.cache_path.exists():
            for record in _load_cache_records(self.cache_path):
                self._cache[record["key"]] = record["response"]
        self._http = httpx.AsyncClient(timeout=self.timeout)

    @classmethod
    def openrouter(cls, model: str, **kw: Any) -> ChatClient:
        """Use OpenRouter, resolving OPENROUTER_API_KEY at request time."""
        return cls(
            endpoint=Endpoint(
                OPENROUTER_BASE_URL,
                model,
                api_key_env="OPENROUTER_API_KEY",
                extra_params=kw.pop("extra_params", None),
            ),
            **kw,
        )

    @classmethod
    def anthropic(cls, model: str, **kw: Any) -> ChatClient:
        """Use Anthropic Messages, resolving ANTHROPIC_API_KEY at request time."""
        return cls(
            endpoint=Endpoint(
                ANTHROPIC_BASE_URL,
                model,
                api_key_env="ANTHROPIC_API_KEY",
                provider="anthropic",
                extra_params=kw.pop("extra_params", None),
            ),
            **kw,
        )

    @classmethod
    def openai(cls, model: str, **kw: Any) -> ChatClient:
        """Use OpenAI, resolving OPENAI_API_KEY at request time."""
        return cls(
            endpoint=Endpoint(
                OPENAI_BASE_URL,
                model,
                api_key_env="OPENAI_API_KEY",
                extra_params=kw.pop("extra_params", None),
            ),
            **kw,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    @staticmethod
    def _key(payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    async def chat(
        self,
        payload: dict[str, Any],
        *,
        cache_salt: str | None = None,
    ) -> dict[str, Any]:
        """POST chat; cache_salt affects only cache identity, never the wire body."""
        return await self._post("/chat/completions", payload, cache_salt)

    async def completions(
        self,
        payload: dict[str, Any],
        *,
        cache_salt: str | None = None,
    ) -> dict[str, Any]:
        """POST raw-text completions with the same explicit cache contract."""
        return await self._post("/completions", payload, cache_salt)

    async def _post(
        self,
        route: str,
        payload: dict[str, Any],
        cache_salt: str | None = None,
    ) -> dict[str, Any]:
        if self.cache != "off" and cache_salt is None:
            raise ValueError("cached ChatClient calls require an explicit cache_salt")
        if "model" in payload:
            raise ValueError("payload must not include 'model'; set Endpoint.model instead")
        payload = {"model": self.endpoint.model, **(self.endpoint.extra_params or {}), **payload}
        if self.endpoint.provider == "anthropic":
            if route != "/chat/completions":
                raise UnsupportedRequestError(f"route {route!r} is not supported on anthropic")
            url = self.endpoint.base_url.rstrip("/") + "/v1/messages"
            body = to_anthropic(payload)
        else:
            url = self.endpoint.base_url.rstrip("/") + route
            body = payload
        key = None
        if self.cache != "off":
            key = self._key({"route": route, "payload": payload, "cache_salt": cache_salt})
            if key in self._cache:
                return self._cache[key]

        delay = 1.0
        last_error = ""
        async with self._sem:
            for attempt in range(self.max_retries):
                send_body = body
                if self._use_max_completion_tokens and "max_tokens" in body:
                    send_body = dict(body)
                    send_body["max_completion_tokens"] = send_body.pop("max_tokens")
                headers = self.endpoint.headers()
                try:
                    response = await self._http.post(url, json=send_body, headers=headers)
                except httpx.HTTPError as exc:
                    last_error = _redact(str(exc), headers)
                else:
                    if response.status_code in RETRYABLE_STATUS:
                        last_error = _redact(
                            f"HTTP {response.status_code}: {response.text}", headers
                        )
                    elif (
                        self.endpoint.provider == "openai"
                        and response.status_code == 400
                        and "max_tokens" in send_body
                        and "max_completion_tokens" in response.text
                    ):
                        # Concurrent first requests must each retry their own 400.
                        self._use_max_completion_tokens = True
                        last_error = "server wants max_completion_tokens; retrying"
                        continue
                    elif response.status_code >= 400:
                        raise UnsupportedRequestError(
                            _redact(
                                f"HTTP {response.status_code}: {response.text}",
                                headers,
                            )
                        )
                    else:
                        try:
                            data = response.json()
                        except ValueError:
                            last_error = "non-JSON response body"
                        else:
                            embedded = _embedded_error(data)
                            if embedded is not None:
                                last_error = _redact(f"embedded error: {embedded}", headers)
                            else:
                                if self.endpoint.provider == "anthropic":
                                    data = from_anthropic(data)
                                if key is not None and _has_completion(data):
                                    await self._store(key, data)
                                return data
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
        raise RuntimeError(f"chat request failed after {self.max_retries} retries: {last_error}")

    async def _store(self, key: str, response: dict[str, Any]) -> None:
        async with self._cache_lock:
            if self.cache == "disk" and self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as cache_file:
                    cache_file.write(json.dumps({"key": key, "response": response}) + "\n")
                    cache_file.flush()
                    os.fsync(cache_file.fileno())
            self._cache[key] = response


def _redact(detail: str, headers: dict[str, str]) -> str:
    # Providers and transports may echo credentials in error messages.
    for name in ("Authorization", "x-api-key"):
        secret = headers.get(name, "").removeprefix("Bearer ")
        if secret:
            detail = detail.replace(secret, "[redacted]")
    return detail[:500]


def _has_completion(data: dict[str, Any]) -> bool:
    choices = data.get("choices") or []
    return bool(choices) and all(
        choice.get("text")
        or (choice.get("message") or {}).get("content")
        or (choice.get("message") or {}).get("tool_calls")
        for choice in choices
    )


def _embedded_error(data: dict[str, Any]) -> str | None:
    if data.get("error") or data.get("type") == "error":
        return str(data.get("error") or "provider error")
    for choice in data.get("choices") or []:
        if choice.get("error"):
            return str(choice["error"])
        if choice.get("finish_reason") == "error":
            return "choice finish_reason=error"
    return None


def cached_client(
    endpoint: Endpoint,
    cache_dir: Path,
    tag: str,
    concurrency: int = 32,
    request_semaphore: asyncio.Semaphore | None = None,
) -> ChatClient:
    """Opt into a tagged disk cache; every call still requires cache_salt."""
    return ChatClient(
        endpoint=endpoint,
        concurrency=concurrency,
        cache="disk",
        cache_path=cache_dir / f"cache_{tag}.jsonl",
        request_semaphore=request_semaphore,
    )
