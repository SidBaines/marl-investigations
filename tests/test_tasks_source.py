"""The shipped catalog pins schemas without storing benchmark problem text."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from marli.registry import CATALOG, Registry, catalog_names
from marli.tasks.source import SOURCES, TaskSourceSpec

EXPECTED = {
    "aime_2025": ("MathArena/aime_2025", "train", "integer", "cc-by-nc-sa-4.0"),
    "aime_2026": ("MathArena/aime_2026", "train", "integer", "cc-by-nc-sa-4.0"),
    "hmmt_feb_2025": ("MathArena/hmmt_feb_2025", "train", "latex", "cc-by-nc-sa-4.0"),
    "hmmt_feb_2026": ("MathArena/hmmt_feb_2026", "train", "latex", "cc-by-nc-sa-4.0"),
    "beyond_aime": ("ByteDance-Seed/BeyondAIME", "test", "integer", "cc0-1.0"),
    "imo_answerbench": ("OpenEvals/IMO-AnswerBench", "train", "latex", "cc-by-4.0"),
    "math500": ("HuggingFaceH4/MATH-500", "test", "latex", "mit"),
    "polaris_53k": ("POLARIS-Project/Polaris-Dataset-53K", "train", "latex", "apache-2.0"),
    "deepmath_103k": ("zwhe99/DeepMath-103K", "train", "latex", "mit"),
    "dapo_math_17k": ("BytedTsinghua-SIA/DAPO-Math-17k", "train", "integer", "apache-2.0"),
}


def test_catalog_entries() -> None:
    assert SOURCES.names() == sorted(EXPECTED)
    assert CATALOG["tasks"] == "marli.tasks.source:SOURCES"
    assert catalog_names("tasks") == sorted(EXPECTED)
    for name, spec in SOURCES.load_all().items():
        assert (spec.hf_id, spec.split, spec.answer_format, spec.license) == EXPECTED[name]
        assert spec.name == name
        assert spec.kind == "math"
        assert spec.commit_text is False
        assert spec.config is None


def test_nested_paths_and_filtering_are_explicit() -> None:
    dapo = SOURCES.load("dapo_math_17k")
    assert dapo.fields["id"] == "extra_info.index"
    assert dapo.prompt_path == "prompt.0.content"
    assert dapo.answer_path == "reward_model.ground_truth"
    assert dapo.dedupe is True
    assert SOURCES.load("hmmt_feb_2026").filters == {"answer_nonempty": True}
    assert "unverified" in SOURCES.load("math500").notes


def test_frozen_and_conservative_defaults() -> None:
    spec = TaskSourceSpec(
        "synthetic", "local", "train", "math", {"prompt": "p", "answer": "a"}, "latex", "mit"
    )
    assert spec.commit_text is False
    assert spec.filters == {}
    assert spec.dedupe is False
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "unsupported"},
        {"answer_format": "float"},
        {"fields": {"answer": "answer"}},
        {"fields": {"prompt": "problem"}},
        {"fields": {"prompt": "problem", "answer": "answer", "typo": "column"}},
        {"fields": {"prompt": "", "answer": "answer"}},
        {"filters": {"unknown": True}},
        {"filters": {"answer_nonempty": "true"}},
        {"commit_text": "false"},
        {"dedupe": "false"},
    ],
)
def test_invalid_source_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(SOURCES.load("aime_2025"), **changes)


def test_unknown_yaml_key_rejected(tmp_path: Path) -> None:
    original = SOURCES.path("aime_2025").read_text()
    (tmp_path / "aime_2025.yaml").write_text(original + "typo: true\n")
    with pytest.raises(ValueError, match="unknown keys.*typo"):
        Registry("tasks", tmp_path, TaskSourceSpec).load("aime_2025")
