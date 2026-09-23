"""Share judge transport so each classifier owns only its rubric and parser.

Ported from scimt's src/scimt/utils/judge.py. Transport exhaustion returns
None: callers must record a failed judgment rather than inventing a label.
"""

from __future__ import annotations

import asyncio
import os
import warnings
from typing import Any

import httpx

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def judge_headers() -> dict[str, str]:
    """Request headers for the Anthropic Messages API (reads ANTHROPIC_API_KEY)."""
    return {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


async def anthropic_judge(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    headers: dict[str, str],
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 8,
    temperature: float | None = None,
) -> str | None:
    """Return raw judge text, or None after four attempts with backoff."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if temperature is not None:
        body["temperature"] = temperature
    async with sem:
        for attempt in range(4):
            status: int | None = None
            try:
                response = await client.post(
                    ANTHROPIC_URL,
                    json=body,
                    headers=headers,
                    timeout=60,
                )
                status = response.status_code
                response.raise_for_status()
                return response.json()["content"][0]["text"]
            except Exception:
                if attempt == 3:
                    warnings.warn(
                        f"judge request failed after 4 attempts; last status: {status}",
                        stacklevel=2,
                    )
                    return None
                await asyncio.sleep(2 * (attempt + 1))
    return None
