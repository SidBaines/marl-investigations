"""Policy contracts: token-level samplers and chat-level (API) policies.

Two kinds of seat exist:

- **Token policies** (``TokenPolicy``) take prompt token ids and return sampled
  ids + logprobs. Backends: Tinker ``SamplingClient`` (cloud) and a vLLM
  server (``/v1/completions`` with a token-id prompt, ``return_token_ids``).
  Only token policies can be trained, and trainable seats must sample the raw
  policy distribution (``check_trainable_sampling``) because both backends
  return raw logprobs regardless of temperature/top-k — anything else would be
  silently off-policy.
- **Chat policies** (``ChatPolicy``) take messages and return text + tool calls
  (OpenAI/Anthropic/OpenRouter through ``marli.llm.client``). They are frozen
  by construction — they can play partner/judge/teacher roles, never learners
  — and their token counts are in a foreign tokenizer, tagged in ``Usage``.

``PolicyRef`` strings name policies in configs and on the CLI; see
``policy/refs.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from marli.errors import ConfigError
from marli.interact.types import Termination, Usage
from marli.render.base import Msg, ParsedToolCall, ToolSpec


@dataclass(frozen=True)
class SamplingSpec:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 = disabled (Tinker/vLLM convention)
    # the renderer's stop ids; included in the returned completion
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ConfigError(f"SamplingSpec.max_tokens must be positive, got {self.max_tokens}")


def check_trainable_sampling(spec: SamplingSpec, *, policy_id: str) -> None:
    """Raise if a trainable seat would sample anything but the raw distribution."""
    if spec.temperature != 1.0 or spec.top_p != 1.0 or spec.top_k != -1:
        raise ConfigError(
            f"trainable policy {policy_id!r} must sample with temperature=1, top_p=1, top_k=-1 "
            f"(got {spec.temperature}, {spec.top_p}, {spec.top_k}): both backends return raw "
            "logprobs, so truncated/tempered sampling would make RL silently off-policy"
        )


@dataclass(frozen=True)
class Sample:
    # sampled ids, including the stop token when one was sampled
    completion_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None  # aligned with completion_ids
    # STOP or LENGTH (backend view; the renderer may refine to MALFORMED)
    termination: Termination
    policy_version: int | None
    usage: Usage = Usage()


@dataclass(frozen=True)
class CallMeta:
    """Who is calling — for logging, scripted test policies and per-agent routing.

    Backends must not change *what* they sample based on this (except scripted
    policies, which exist to be deterministic functions of it)."""

    episode_id: str = ""
    agent_id: str = ""
    role: str = ""
    call_index: int = 0  # 0-based index of this call among the agent's calls
    purpose: str = "act"  # interact.types.Purpose value
    # Completion ids of this same call already at the end of prompt_ids: an episode
    # continuation extends a recorded, budget-cut completion (eval continue).
    continued_tokens: int = 0


@runtime_checkable
class TokenPolicy(Protocol):
    policy_id: str
    trainable: bool
    renderer_name: str  # must match the agent's renderer (validated at seating time)

    async def sample(
        self,
        prompt_ids: Sequence[int],
        spec: SamplingSpec,
        *,
        seed: int,
        meta: CallMeta | None = None,
    ) -> Sample: ...


@dataclass(frozen=True)
class ChatReply:
    content: str
    thinking: str | None
    tool_calls: tuple[ParsedToolCall, ...]
    termination: Termination
    usage: Usage = Usage()
    # provider response (for audit), never secrets
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ChatPolicy(Protocol):
    policy_id: str
    trainable: bool  # always False

    async def chat(
        self,
        messages: Sequence[Msg],
        tools: Sequence[ToolSpec],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        # REQUIRED and per call: identical prompts must not collapse to one sample.
        cache_salt: str,
        meta: CallMeta | None = None,
    ) -> ChatReply: ...


Policy = TokenPolicy | ChatPolicy
