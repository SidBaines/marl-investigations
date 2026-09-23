"""Keep API seats frozen and preserve tool identities, usage and sampling salts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from marli.budget import SpendGuard
from marli.errors import BackendError, BudgetExceededError, ConfigError
from marli.interact.types import Termination, Usage
from marli.llm.client import OPENROUTER_BASE_URL, ChatClient, usage_of
from marli.policy.base import CallMeta, ChatReply
from marli.render.base import Msg, ParsedToolCall, ToolSpec
from marli.seeds import cache_salt as stable_id


def _messages(messages: Sequence[Msg]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    pending: list[tuple[str, str]] = []
    for index, message in enumerate(messages):
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "assistant":
            pending.clear()
        if message.role == "assistant" and message.tool_calls:
            calls = []
            for call_index, call in enumerate(message.tool_calls):
                arguments = json.dumps(call.arguments, sort_keys=True)
                call_id = call.id or f"call_{stable_id(index, call_index, call.name, arguments)}"
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": arguments},
                    }
                )
                pending.append((call.name, call_id))
            item["tool_calls"] = calls
        elif message.role == "tool":
            call_id = message.tool_call_id
            for pending_index, (name, candidate_id) in enumerate(pending):
                if (call_id is not None and candidate_id == call_id) or (
                    call_id is None and (message.name is None or message.name == name)
                ):
                    call_id = candidate_id
                    pending.pop(pending_index)
                    break
            if call_id is None:
                raise ConfigError("API tool result has no tool_call_id or matching assistant call")
            item["tool_call_id"] = call_id
        converted.append(item)
    return converted


def _tool_call(call: dict[str, Any]) -> ParsedToolCall:
    function = call.get("function") or {}
    name = function.get("name")
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (ValueError, TypeError):
        arguments = None
    if not isinstance(arguments, dict):
        arguments = None
    return ParsedToolCall(
        name=name,
        arguments=arguments,
        raw=json.dumps(call),
        ok=isinstance(name, str) and bool(name) and arguments is not None,
        id=call.get("id"),
    )


class APIPolicy:
    """Prices map ``provider/model`` to input/output USD per million tokens.

    OpenRouter reports cost directly. Other providers require a price entry
    before a finite spend guard can safely admit a call.
    """

    trainable: bool = False

    def __init__(
        self,
        policy_id: str,
        client: ChatClient,
        *,
        provider: str | None = None,
        spend: SpendGuard | None = None,
        api_prices: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.policy_id = policy_id
        self.client = client
        self.provider = provider or (
            "openrouter"
            if client.endpoint.base_url.rstrip("/") == OPENROUTER_BASE_URL
            else client.endpoint.provider
        )
        self.spend = spend
        self.api_prices = api_prices or {}
        model_key = f"{self.provider}/{client.endpoint.model}"
        if (
            spend is not None
            and spend.remaining() is not None
            and self.provider != "openrouter"
            and model_key not in self.api_prices
        ):
            raise ConfigError(f"Finite API spend guard requires a price entry for {model_key}")

    async def chat(
        self,
        messages: Sequence[Msg],
        tools: Sequence[ToolSpec],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        cache_salt: str,
        meta: CallMeta | None = None,
    ) -> ChatReply:
        if not cache_salt:
            raise ConfigError("APIPolicy requires a non-empty cache_salt for every call")
        payload: dict[str, Any] = {
            "messages": _messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.provider in ("openai", "openrouter"):
            payload["seed"] = seed
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        model_key = f"{self.provider}/{self.client.endpoint.model}"
        price = self.api_prices.get(model_key)
        if self.spend is not None and price is not None:
            self.spend.check(max_tokens * price["output"] / 1_000_000, self.policy_id)
        try:
            response = await self.client.chat(payload, cache_salt=cache_salt)
        except (BudgetExceededError, ConfigError):
            raise
        except (RuntimeError, ValueError) as exc:
            raise BackendError(f"API chat failed: {exc}") from exc
        usage = usage_of(response)
        prompt_tokens = int(usage["prompt_tokens"] or 0)
        completion_tokens = int(usage["completion_tokens"] or 0)
        cost = usage["cost_usd"]
        if cost is None and price is not None:
            cost = (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1e6
        if cost is None and self.spend is not None and self.spend.remaining() is not None:
            raise BackendError(f"API response for {model_key} has no cost or configured price")
        cost = float(cost or 0.0)
        if self.spend is not None:
            self.spend.charge(cost, self.policy_id)
        try:
            choice = response["choices"][0]
            message = choice["message"]
            calls = tuple(_tool_call(call) for call in message.get("tool_calls") or ())
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise BackendError("API returned an invalid chat response") from exc
        return ChatReply(
            content=message.get("content") or "",
            thinking=message.get("reasoning_content", message.get("reasoning")),
            tool_calls=calls,
            termination={
                "stop": Termination.STOP,
                "tool_calls": Termination.STOP,
                "length": Termination.LENGTH,
                "content_filter": Termination.ERROR,
            }.get(choice.get("finish_reason"), Termination.ERROR),
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=(
                    None
                    if usage["cached_prompt_tokens"] is None
                    else int(usage["cached_prompt_tokens"])
                ),
                cost_usd=cost,
                tokenizer=f"api:{model_key}",
            ),
            raw=response,
        )
