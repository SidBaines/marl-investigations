"""A LoRA adapter on a marli-served vLLM inherits the server's registered base model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marli.errors import ConfigError
from marli.eval.policies import PolicySpec, resolve_spec


def test_adapter_on_marli_server_inherits_base_model(tmp_path: Path) -> None:
    server = tmp_path / "server.json"
    server.write_text(json.dumps({"kind": "server", "hf_id": "Qwen/Qwen3.8-27B"}))
    _, model, renderer = resolve_spec(PolicySpec(f"vllm:@{server}#exp2-s29-opener"))
    assert model is not None and model.name == "qwen3_8_27b"
    assert renderer == model.renderer
    # The served base name also resolves, and an explicit model still wins.
    _, base, _ = resolve_spec(PolicySpec(f"vllm:@{server}#qwen3_8_27b"))
    assert base is not None and base.name == "qwen3_8_27b"
    _, explicit, _ = resolve_spec(PolicySpec(f"vllm:@{server}#exp2-s29-opener", model="qwen3_8b"))
    assert explicit is not None and explicit.name == "qwen3_8b"


def test_unreadable_or_foreign_server_keeps_the_old_error(tmp_path: Path) -> None:
    for content in ("not json", json.dumps({"kind": "server"})):
        server = tmp_path / "server.json"
        server.write_text(content)
        with pytest.raises(ConfigError, match="needs a renderer"):
            resolve_spec(PolicySpec(f"vllm:@{server}#exp2-s29-opener"))
    with pytest.raises(ConfigError, match="needs a renderer"):
        resolve_spec(PolicySpec(f"vllm:@{tmp_path / 'missing.json'}#adapter"))
