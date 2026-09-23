"""Factories share tokenizer bytes, never the per-lineage renderer state.

Importing this registry (including selecting ``fake``) needs no transformers
installation. New families register a factory accepting an optional HF id.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache, partial
from typing import TYPE_CHECKING

from marli.errors import ConfigError
from marli.render.base import DeltaRenderer
from marli.render.fake import FakeRenderer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@cache
def _tokenizer(hf_id: str) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id)


def _fake(hf_id: str | None) -> DeltaRenderer:
    return FakeRenderer()


def _qwen(name: str, hf_id: str | None) -> DeltaRenderer:
    from marli.render.hf import HFTemplateRenderer
    from marli.render.qwen import qwen_profile

    xml = name.startswith("qwen3_5")
    tokenizer = _tokenizer(hf_id or ("Qwen/Qwen3.5-4B" if xml else "Qwen/Qwen3-8B"))
    profile = qwen_profile(tokenizer, xml=xml, thinking=not name.endswith("_nothink"))
    return HFTemplateRenderer(tokenizer, name, profile)


_FACTORIES: dict[str, Callable[[str | None], DeltaRenderer]] = {
    "fake": _fake,
    "qwen3_5": partial(_qwen, "qwen3_5"),
    "qwen3_5_nothink": partial(_qwen, "qwen3_5_nothink"),
    "qwen3": partial(_qwen, "qwen3"),
    "qwen3_nothink": partial(_qwen, "qwen3_nothink"),
}


def get_renderer(name: str, *, hf_id: str | None = None) -> DeltaRenderer:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ConfigError(
            f"Unknown renderer {name!r}; valid names: {', '.join(sorted(_FACTORIES))}"
        ) from None
    return factory(hf_id)
