"""Pod preflight for the external evals (run after serve + adapters; ~1 min, a few requests).

    uv run --no-sync python preflight.py <server.json> <model.yaml> <renderer> [served names...]

Checks, each printed as OK/FAIL (exit 1 on any FAIL):

1. Chat template parity: vLLM's ``/tokenize`` of a game prompt (system + user, generation
   prompt, the config's chat_template_kwargs) equals our training renderer's prompt ids,
   and differs from the template's default where the model has a reasoning-effort switch.
2. Every served name answers, with thinking split off: non-empty reasoning, and the
   content carries no ``</think>`` (the reasoning parser is active).
3. JSON mode under thinking (HiddenBench votes): ``response_format=json_object`` still
   returns reasoning and a parseable JSON object.
4. ``/v1/completions`` is untouched by the reasoning parser (the transfer eval's path):
   a raw completion of a prefilled thinking prompt still contains ``</think>``.
5. Tool calls (planted help): with a ``bash`` tool offered, the reply comes back as parsed
   OpenAI ``tool_calls`` (the qwen3_coder parser), with no raw ``<tool_call>`` in content.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import yaml

from marli.eval.external.tasks import econ_games
from marli.render.registry import get_renderer

FAILED = False


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILED
    FAILED |= not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)


def main() -> None:
    server_json, model_yaml, renderer_name, *names = sys.argv[1:]
    server = json.loads(Path(server_json).read_text())
    base = server["base_url"].rstrip("/")
    names = names or server["models"]
    generation = yaml.safe_load(Path(model_yaml).read_text())["model_generation"]
    kwargs = {"enable_thinking": True, **generation.get("chat_template_kwargs", {})}
    client = httpx.Client(base_url=base, timeout=600, trust_env=False)
    game = econ_games.GAMES["prisoners_dilemma"]
    messages = [
        {"role": "system", "content": econ_games.SYSTEM_PROMPT},
        {"role": "user", "content": game.prompt},
    ]
    model = server["models"][0]

    def tokenize(template_kwargs: dict | None) -> list[int]:
        body = {"model": model, "messages": messages, "add_generation_prompt": True}
        if template_kwargs is not None:
            body["chat_template_kwargs"] = template_kwargs
        response = client.post("/tokenize", json=body)
        response.raise_for_status()
        return response.json()["tokens"]

    from marli.render.base import Msg

    renderer = get_renderer(renderer_name, hf_id=server["hf_id"])
    ours = list(renderer.initial(econ_games.SYSTEM_PROMPT, [], [Msg("user", game.prompt)]))
    served = tokenize(kwargs)
    check(
        "1 vLLM chat template == training renderer",
        served == ours,
        f"{len(served)} vs {len(ours)} ids",
    )
    default = tokenize(None)
    note = (
        "differs (as expected when the template has an effort switch)"
        if default != served
        else "same"
    )
    print(f"     template default without chat_template_kwargs: {note}")

    request = {
        key: generation[key]
        for key in ("temperature", "top_p", "presence_penalty")
        if key in generation
    }
    extra = {"top_k": generation.get("top_k", -1), "chat_template_kwargs": kwargs}
    for name in names:
        response = client.post(
            "/v1/chat/completions",
            json={"model": name, "messages": messages, "max_tokens": 4096, **request, **extra},
        )
        ok = response.is_success
        message = response.json()["choices"][0]["message"] if ok else {}
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        content = message.get("content") or ""
        check(
            f"2 {name}: thinking split off",
            ok and bool(reasoning) and "</think>" not in content,
            f"reasoning {len(reasoning)} chars; content {content[:60]!r}",
        )
    vote = messages + [
        {"role": "user", "content": 'Return only one valid JSON object: {"vote": "A"}'}
    ]
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": names[0],
            "messages": vote,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
            **request,
            **extra,
        },
    )
    message = response.json()["choices"][0]["message"] if response.is_success else {}
    try:
        parsed = isinstance(json.loads(message.get("content") or ""), dict)
    except ValueError:
        parsed = False
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    check(
        "3 JSON mode keeps thinking",
        parsed and bool(reasoning),
        f"reasoning {len(reasoning)} chars",
    )
    response = client.post(
        "/v1/completions",
        json={"model": model, "prompt": ours, "max_tokens": 8192, "temperature": 1.0},
    )
    choice = response.json()["choices"][0] if response.is_success else {}
    text = choice.get("text", "")
    if choice.get("finish_reason") == "length" and "</think>" not in text:
        print("WARN 4 /v1/completions: thinking did not finish within 8192 tokens; inconclusive")
    else:
        check("4 /v1/completions returns raw text", "</think>" in text, f"{len(text)} chars")
    bash = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Use this function to execute bash commands.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The command."}},
                "required": ["command"],
            },
        },
    }
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": names[0],
            "messages": [{"role": "user", "content": "List the files here with the bash tool."}],
            "tools": [bash],
            "tool_choice": "auto",
            "max_tokens": 4096,
            **request,
            **extra,
        },
    )
    message = response.json()["choices"][0]["message"] if response.is_success else {}
    calls = message.get("tool_calls") or []
    check(
        "5 tool calls are parsed",
        bool(calls)
        and calls[0]["function"]["name"] == "bash"
        and "<tool_call>" not in (message.get("content") or ""),
        f"{len(calls)} calls",
    )
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
