"""Tokenizer loading and fingerprinting are shared across agent lineages."""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace

import pytest

from marli.render import registry


def test_tokenizer_and_fingerprint_cached_per_hf_id(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[str] = []
    vocab_reads: list[str] = []

    @cache
    def tokenizer(hf_id: str) -> SimpleNamespace:
        loads.append(hf_id)
        vocab = {"<|im_end|>": 1, "<|endoftext|>": 2, hf_id: 3}

        def get_vocab() -> dict[str, int]:
            vocab_reads.append(hf_id)
            return vocab

        return SimpleNamespace(
            get_vocab=get_vocab,
            chat_template="template",
            added_tokens_decoder={
                1: SimpleNamespace(content="<|im_end|>", special=True),
                2: SimpleNamespace(content="<|endoftext|>", special=True),
            },
            convert_tokens_to_ids=vocab.__getitem__,
        )

    monkeypatch.setattr(registry, "_tokenizer", tokenizer)
    monkeypatch.setattr(registry, "_tokenizer_sha", cache(registry._tokenizer_sha.__wrapped__))
    first = registry.get_renderer("qwen3", hf_id="local/one")
    second = registry.get_renderer("qwen3_nothink", hf_id="local/one")
    third = registry.get_renderer("qwen3", hf_id="local/two")
    assert first is not second and second is not third
    assert first.tokenizer is second.tokenizer
    assert first.tokenizer_sha == second.tokenizer_sha != third.tokenizer_sha
    assert len(first.tokenizer_sha) == 16
    int(first.tokenizer_sha, 16)
    assert loads == vocab_reads == ["local/one", "local/two"]
