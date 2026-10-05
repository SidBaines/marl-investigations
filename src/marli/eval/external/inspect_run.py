"""Run one Inspect suite cell through Inspect's Python API (imported lazily, [external]).

The model is Inspect's generic OpenAI-compatible provider (``openai-api``) pointed at
the vLLM server, so requests are plain chat completions rendered by the model's own
chat template. vLLM-specific fields (``top_k``, ``chat_template_kwargs``, ...) travel
in ``extra_body``; Inspect sends nothing else on its own. Inspect's console display is
switched off and anything it might print is sent to stderr: the CLI's stdout is one
JSON line.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

from marli.errors import BackendError
from marli.eval.external.serving import Endpoint, request_fields
from marli.eval.external.spec import ExternalSuite
from marli.registry import FnRegistry

_TASKS: FnRegistry = FnRegistry("external inspect tasks")
# Tests only: scripted mockllm players by name (``inspect:mockllm/<name>``), each an Inspect
# ``custom_outputs`` callable. An unregistered name uses mockllm's default reply.
MOCK_PLAYERS: dict[str, Any] = {}


async def run_cell(
    suite: ExternalSuite,
    endpoint: Endpoint,
    settings: dict[str, Any],
    *,
    task_args: dict[str, Any],
    epochs: int,
    limit: int | None,
    log_dir: Path,
    max_connections: int,
    max_error_rate: float,
    request_timeout_s: int = 1800,
) -> list[Path]:
    """Evaluate the suite's task against one endpoint; return its ``.eval`` logs."""
    from inspect_ai import eval_async
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_ai.util._display import display_type_initialized, init_display_type

    if not display_type_initialized():
        init_display_type("none")
    assert suite.task is not None
    task = _TASKS.get(suite.task)(**task_args)
    request = request_fields(settings)
    config = GenerateConfig(
        max_connections=max_connections,
        timeout=request_timeout_s,
        temperature=request.get("temperature"),
        top_p=request.get("top_p"),
        max_tokens=request.get("max_tokens"),
        presence_penalty=request.get("presence_penalty"),
        frequency_penalty=request.get("frequency_penalty"),
        seed=request.get("seed"),
        extra_body=request.get("extra_body"),
    )
    if endpoint.provider == "mockllm":
        players = (
            {"custom_outputs": MOCK_PLAYERS[endpoint.model]}
            if endpoint.model in MOCK_PLAYERS
            else {}
        )
        model = get_model(f"mockllm/{endpoint.model}", config=config, memoize=False, **players)
    else:
        model = get_model(
            f"openai-api/marli/{endpoint.model}",
            base_url=endpoint.api_base,
            api_key="EMPTY",  # vLLM servers here run without --api-key
            config=config,
            memoize=False,
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        logs = await eval_async(
            task,
            model=model,
            log_dir=str(log_dir),
            log_format="eval",
            epochs=epochs,
            limit=limit,
            fail_on_error=max_error_rate,
            score=True,
        )
    paths = []
    for log in logs:
        if log.status != "success":
            message = log.error.message if log.error is not None else log.status
            raise BackendError(f"Inspect eval of {endpoint.model!r} ended {log.status}: {message}")
        paths.append(Path(log.location))
    return paths
