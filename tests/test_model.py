"""Registry data and validation must preserve unambiguous backend selection."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
from datetime import date
from itertools import product
from pathlib import Path

import pytest
import yaml

import marli.model as model_module
from marli.errors import ConfigError
from marli.model import (
    MODELS,
    ModelSpec,
    TinkerPrices,
    for_hf_id,
    list_models,
    load_model,
    tinker_models,
)
from marli.registry import Registry, catalog_kinds, catalog_names


@pytest.mark.parametrize(
    "name,hf_id,family,architecture,ctx,prices,local,thinking,tool_format",
    [
        (
            "qwen3_5_4b",
            "Qwen/Qwen3.5-4B",
            "qwen3_5",
            "Qwen3_5ForConditionalGeneration",
            65536,
            (0.33, 1.01, 0.74),
            "unverified",
            True,
            "qwen3_5_xml",
        ),
        (
            "qwen3_5_9b",
            "Qwen/Qwen3.5-9B",
            "qwen3_5",
            "Qwen3_5ForConditionalGeneration",
            65536,
            (0.66, 2.00, 1.46),
            "unverified",
            True,
            "qwen3_5_xml",
        ),
        (
            "qwen3_5_9b_base",
            "Qwen/Qwen3.5-9B-Base",
            "qwen3_5",
            "Qwen3_5ForConditionalGeneration",
            65536,
            (0.66, 2.00, 1.46),
            "unverified",
            False,
            "qwen3_5_xml",
        ),
        (
            "qwen3_6_35b_a3b",
            "Qwen/Qwen3.6-35B-A3B",
            "qwen3_5",
            "Qwen3_5MoeForConditionalGeneration",
            65536,
            (0.54, 1.34, 1.18),
            "no",
            True,
            "qwen3_5_xml",
        ),
        (
            "qwen3_5_397b_a17b",
            "Qwen/Qwen3.5-397B-A17B",
            "qwen3_5",
            "Qwen3_5MoeForConditionalGeneration",
            65536,
            (3.00, 7.50, 6.60),
            "no",
            True,
            "qwen3_5_xml",
        ),
        (
            "gpt_oss_20b",
            "openai/gpt-oss-20b",
            "gpt_oss",
            "GptOssForCausalLM",
            32768,
            (0.18, 0.45, 0.40),
            "unverified",
            True,
            "harmony",
        ),
        (
            "gpt_oss_120b",
            "openai/gpt-oss-120b",
            "gpt_oss",
            "GptOssForCausalLM",
            32768,
            (0.33, 0.84, 0.74),
            "no",
            True,
            "harmony",
        ),
        (
            "qwen3_8b",
            "Qwen/Qwen3-8B",
            "qwen3",
            "Qwen3ForCausalLM",
            32768,
            (0.20, 0.60, 0.44),
            "unverified",
            True,
            "qwen3_json",
        ),
        (
            "qwen3_4b_instruct_2507",
            "Qwen/Qwen3-4B-Instruct-2507",
            "qwen3",
            "Qwen3ForCausalLM",
            None,
            None,
            "unverified",
            False,
            "qwen3_json",
        ),
    ],
)
def test_published_model_entries(
    name: str,
    hf_id: str,
    family: str,
    architecture: str,
    ctx: int | None,
    prices: tuple[float, float, float] | None,
    local: str,
    thinking: bool,
    tool_format: str,
) -> None:
    model = load_model(name)
    assert model.name == MODELS.path(name).stem == name
    assert model.hf_id == hf_id
    assert model.family == family
    assert model.renderer == {"gpt_oss": "gpt_oss_medium"}.get(family, family)
    assert model.architecture == architecture
    assert model.max_ctx == 32768
    assert model.default_max_tokens == (8192 if thinking else 2048)
    assert model.thinking is thinking
    assert model.tool_format == tool_format
    assert model.local == local
    assert model.tinker_max_ctx == ctx
    assert model.tinker_id == (hf_id if prices is not None else None)
    assert model.tinker_prices == (TinkerPrices(*prices, "2026-09-23") if prices else None)
    assert for_hf_id(hf_id) == model
    if family == "qwen3_5" and local == "unverified":
        assert model.notes == (
            "Qwen3.5 is a vision-language architecture; "
            "local LoRA support to be verified (M2 spike)"
        )
    if family == "qwen3_5" and local == "no":
        assert model.notes == (
            "Qwen3.5 vision-language MoE architecture; not supported by the local learner"
        )
    if prices is None:
        assert "Retired on Tinker 2026-06-12" in model.notes


def test_all_entries_load_and_tinker_subset_excludes_retired_model() -> None:
    names = list_models()
    assert len(names) == 9
    assert names == sorted(names)
    assert [load_model(name).name for name in names] == names
    assert [model.name for model in tinker_models()] == [
        name for name in names if name != "qwen3_4b_instruct_2507"
    ]


def test_model_registry_is_discoverable_in_catalog() -> None:
    assert "models" in catalog_kinds()
    assert catalog_names("models") == list_models()


def test_missing_names_and_hf_ids_raise_config_error() -> None:
    with pytest.raises(ConfigError, match="unknown models.*missing"):
        load_model("missing")
    with pytest.raises(ConfigError, match="no model.*missing/model"):
        for_hf_id("missing/model")


def test_hf_id_twins_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = load_model("qwen3_8b")
    for name in ("first", "second"):
        data = asdict(replace(original, name=name))
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(model_module, "MODELS", Registry("models", tmp_path, ModelSpec))
    with pytest.raises(ConfigError, match="duplicate HF id.*Qwen/Qwen3-8B.*first.*second"):
        for_hf_id(original.hf_id)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"name": "wrong"}, "filename stem"),
        ({"unknown": 1}, "unknown keys.*unknown"),
        ({"family": "wrong"}, "family"),
        ({"tinker_id": None}, "all set or all None"),
        (
            {
                "tinker_prices": {
                    "prefill": 1,
                    "sample": 2,
                    "train": 3,
                    "as_of": "2026-09-23",
                    "typo": 4,
                }
            },
            "invalid tinker_prices.*typo",
        ),
    ],
)
def test_invalid_yaml_is_config_error(
    tmp_path: Path, changes: dict[str, object], match: str
) -> None:
    data = asdict(load_model("qwen3_8b")) | changes
    (tmp_path / "qwen3_8b.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        Registry("models", tmp_path, ModelSpec).load("qwen3_8b")


@pytest.mark.parametrize("field", ["family", "renderer", "tool_format", "local"])
def test_invalid_enum_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field) as caught:
        replace(load_model("qwen3_8b"), **{field: "unsupported"})
    assert "got 'unsupported'" in str(caught.value)


@pytest.mark.parametrize("field", ["max_ctx", "default_max_tokens", "tinker_max_ctx"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "32768"])
def test_context_and_token_caps_are_positive_integers(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a positive integer") as caught:
        replace(load_model("qwen3_8b"), **{field: value})
    assert f"got {value!r}" in str(caught.value)


@pytest.mark.parametrize("max_ctx", [1, 32768])
def test_context_cap_can_equal_tinker_limit(max_ctx: int) -> None:
    assert replace(load_model("qwen3_8b"), max_ctx=max_ctx).max_ctx == max_ctx


def test_context_cap_must_fit_tinker_limit() -> None:
    with pytest.raises(ValueError, match="max_ctx must be <= tinker_max_ctx") as caught:
        replace(load_model("qwen3_8b"), max_ctx=32769)
    assert "32768" in str(caught.value)
    assert "got 32769" in str(caught.value)


def test_non_tinker_model_has_no_tinker_context_limit() -> None:
    model = replace(load_model("qwen3_4b_instruct_2507"), max_ctx=65536)
    assert model.tinker_max_ctx is None
    assert model.max_ctx == 65536


@pytest.mark.parametrize(
    "present",
    [flags for flags in product((False, True), repeat=3) if any(flags) and not all(flags)],
)
def test_partial_tinker_configuration(present: tuple[bool, bool, bool]) -> None:
    original = load_model("qwen3_8b")
    changes = {
        field: getattr(original, field) if keep else None
        for field, keep in zip(
            ("tinker_id", "tinker_max_ctx", "tinker_prices"), present, strict=True
        )
    }
    with pytest.raises(ValueError, match="all set or all None") as caught:
        replace(original, **changes)
    assert f"got {tuple(changes.values())!r}" in str(caught.value)


def test_prices_mapping_conversion_and_frozen_records() -> None:
    original = load_model("qwen3_8b")
    model = ModelSpec(**asdict(original))
    assert model == original
    assert isinstance(model.tinker_prices, TinkerPrices)
    with pytest.raises(FrozenInstanceError):
        model.local = "yes"
    with pytest.raises(FrozenInstanceError):
        model.tinker_prices.prefill = 0


@pytest.mark.parametrize("prices", [{"typo": 1}, {"prefill": 1}, "not a mapping"])
def test_invalid_prices_raise_value_error(prices: object) -> None:
    with pytest.raises(ValueError, match="tinker_prices") as caught:
        replace(load_model("qwen3_8b"), tinker_prices=prices)
    assert f"got {prices!r}" in str(caught.value)


@pytest.mark.parametrize("field", ["prefill", "sample", "train"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, False, "1", None])
def test_invalid_price_amounts(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field) as caught:
        replace(TinkerPrices(1, 2, 3, "2026-09-23"), **{field: value})
    assert f"got {value!r}" in str(caught.value)


@pytest.mark.parametrize("as_of", ["not a date", "2026-02-30", "20260923", date(2026, 9, 23)])
def test_price_date_must_be_iso_date(as_of: object) -> None:
    with pytest.raises(ValueError) as caught:
        TinkerPrices(1, 2, 3, as_of)
    assert "as_of must be a quoted ISO date string 'YYYY-MM-DD'" in str(caught.value)
    assert f"got {as_of!r}" in str(caught.value)


def test_unquoted_yaml_date_explains_required_quoting(tmp_path: Path) -> None:
    data = asdict(load_model("qwen3_8b"))
    data["tinker_prices"]["as_of"] = date(2026, 9, 23)
    yaml_text = yaml.safe_dump(data)
    assert "as_of: 2026-09-23" in yaml_text
    (tmp_path / "qwen3_8b.yaml").write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        Registry("models", tmp_path, ModelSpec).load("qwen3_8b")
    assert "as_of must be a quoted ISO date string 'YYYY-MM-DD'" in str(caught.value)


def test_thinking_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="thinking") as caught:
        replace(load_model("qwen3_8b"), thinking="false")
    assert "got 'false'" in str(caught.value)
