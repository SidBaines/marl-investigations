"""A fake OpenAI-compatible (vLLM-like) server for CPU tests of external harnesses.

Stdlib only. Serves ``GET /v1/models`` (with vLLM's ``root``/``parent`` fields for LoRA
adapters) and ``POST /v1/chat/completions``. Replies come from a policy per served model
name, keyed on the last user message, so a test can stand in for a cooperative or a
selfish adapter. Every request body is recorded for assertions about the settings a
harness actually sent (sampling, ``chat_template_kwargs``, ``response_format``).

Run standalone for the study's dev-box smoke: ``python tests/_fake_openai.py --port 8765``.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# (content, reasoning), or a whole assistant message dict (e.g. with tool_calls).
Reply = Callable[[str, list[dict[str, Any]]], Any]


def scripted_reply(cooperative: bool) -> Reply:
    """Content and reasoning for a cooperative or selfish stand-in model."""

    def reply(model: str, messages: list[dict[str, Any]]) -> tuple[str, str | None]:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = last if isinstance(last, str) else json.dumps(last)
        reasoning = "Let me weigh the options."
        if "JSON" in text or "json" in text:
            options = re.findall(
                r'"vote": <A string, ([^>]*)>',
                "\n".join(str(m["content"]) for m in messages if m["role"] == "user"),
            )
            choices = [c.strip() for c in options[-1].split(",")] if options else ["x"]
            vote = choices[0] if cooperative else choices[-1]
            return json.dumps({"vote": vote, "rationale": "Based on the shared facts."}), reasoning
        if "[Option A or Option B]" in text:
            return ("I choose **Option A**." if cooperative else "Option B"), reasoning
        if "[0 to 100]" in text:
            return ("I give **50 points** to my partner." if cooperative else "0"), reasoning
        if "Output ONLY the choice" in text:
            # FAIRGAME volunteer's dilemma: strategy2 volunteers.
            labels = re.findall(r"Choose between (\S+) and (\S+?)\.", text)
            if labels:
                return (labels[0][1] if cooperative else labels[0][0]), reasoning
            return "OptionB", reasoning
        return "I have a private fact to share with the group.", reasoning

    return reply


class FakeOpenAIServer:
    """Serve scripted chat completions on 127.0.0.1 in a background thread."""

    def __init__(
        self,
        models: dict[str, Reply] | None = None,
        *,
        roots: dict[str, str] | None = None,
        parent: str = "base-model",
        port: int = 0,
        log: str | None = None,
    ) -> None:
        self.models = models or {"base-model": scripted_reply(True)}
        self.roots = roots or {}
        self.parent = parent
        self.requests: list[dict[str, Any]] = []
        self.log = log
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep test output clean
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                if self.path.rstrip("/") != "/v1/models":
                    self._send(404, {"error": "not found"})
                    return
                data = []
                for name in server.models:
                    entry: dict[str, Any] = {"id": name, "object": "model", "owned_by": "vllm"}
                    if name in server.roots:
                        entry.update(root=server.roots[name], parent=server.parent)
                    else:
                        entry.update(root=name, parent=None)
                    data.append(entry)
                self._send(200, {"object": "list", "data": data})

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path.rstrip("/") != "/v1/chat/completions":
                    self._send(404, {"error": "not found"})
                    return
                with server._lock:
                    server.requests.append(body)
                    if server.log:
                        with open(server.log, "a", encoding="utf-8") as stream:
                            stream.write(json.dumps(body) + "\n")
                model = body.get("model")
                if model not in server.models:
                    self._send(404, {"error": {"message": f"model {model!r} not served"}})
                    return
                reply = server.models[model](model, body.get("messages", []))
                finish = "stop"
                if isinstance(reply, dict):  # a full assistant message, e.g. with tool calls
                    message: dict[str, Any] = reply
                    finish = "tool_calls" if reply.get("tool_calls") else "stop"
                else:
                    content, reasoning = reply
                    message = {"role": "assistant", "content": content}
                    if reasoning is not None:
                        message["reasoning_content"] = reasoning
                self._send(
                    200,
                    {
                        "id": f"chatcmpl-{len(server.requests)}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": model,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                )

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeOpenAIServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cooperative", default="base-model,team-adapter")
    parser.add_argument("--selfish", default="selfish-adapter")
    parser.add_argument("--log", help="append every request body to this JSONL file")
    parser.add_argument(
        "--players", default="", help="name=style,... scripted coding agents (_scripted_agent.py)"
    )
    args = parser.parse_args()
    served: dict[str, Reply] = {}
    for name in filter(None, args.cooperative.split(",")):
        served[name] = scripted_reply(True)
    for name in filter(None, args.selfish.split(",")):
        served[name] = scripted_reply(False)
    if args.players:
        from _scripted_agent import openai_player

        for entry in args.players.split(","):
            name, style = entry.split("=")
            served[name] = openai_player(style)
    with FakeOpenAIServer(served, port=args.port, log=args.log) as fake:
        print(f"fake OpenAI server at {fake.base_url}/v1 serving {sorted(served)}", flush=True)
        threading.Event().wait()
