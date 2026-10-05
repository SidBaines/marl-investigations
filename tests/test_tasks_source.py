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
    "deepcoder": ("agentica-org/DeepCoder-Preview-Dataset", "train", "tests", "mit"),
    "deepcoder_no_lcb": ("agentica-org/DeepCoder-Preview-Dataset", "train", "tests", "mit"),
    "lcb_v6": ("livecodebench/code_generation_lite", "test", "tests", "cc"),
}


def test_catalog_entries() -> None:
    assert SOURCES.names() == sorted(EXPECTED)
    assert CATALOG["tasks"] == "marli.tasks.source:SOURCES"
    assert catalog_names("tasks") == sorted(EXPECTED)
    for name, spec in SOURCES.load_all().items():
        assert (spec.hf_id, spec.split, spec.answer_format, spec.license) == EXPECTED[name]
        assert spec.name == name
        code = {"deepcoder", "deepcoder_no_lcb", "lcb_v6"}
        assert spec.kind == ("code" if name in code else "math")
        assert spec.commit_text is False
        assert spec.config is None


def test_deepcoder_without_livecodebench_differs_only_in_subsets() -> None:
    full, clean = SOURCES.load("deepcoder"), SOURCES.load("deepcoder_no_lcb")
    assert full.subsets == ["primeintellect", "taco", "lcbv5"]
    assert clean.subsets == ["primeintellect", "taco"]
    assert clean.subset_splits == {"primeintellect": "train", "taco": "train"}
    different = {"name", "subsets", "subset_splits", "notes"}
    assert {
        key: value for key, value in vars(clean).items() if key not in different
    } == {key: value for key, value in vars(full).items() if key not in different}


def test_nested_paths_and_filtering_are_explicit() -> None:
    dapo = SOURCES.load("dapo_math_17k")
    assert dapo.fields["id"] == "extra_info.index"
    assert dapo.prompt_path == "prompt.0.content"
    assert dapo.answer_path == "reward_model.ground_truth"
    assert dapo.dedupe is True
    assert dapo.prompt_transform == "dapo_strip_wrapper"
    assert dapo.fields == {"id": "extra_info.index"}
    assert SOURCES.load("hmmt_feb_2026").filters == {"answer_nonempty": True}
    assert "unverified" in SOURCES.load("math500").notes


def test_frozen_and_conservative_defaults() -> None:
    spec = TaskSourceSpec(
        "synthetic", "local", "train", "math", {"prompt": "p", "answer": "a"}, "latex", "mit"
    )
    assert spec.commit_text is False
    assert spec.filters == {}
    assert spec.dedupe is False
    assert spec.prompt_transform is None
    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "unsupported"},
        {"answer_format": "float"},
        {"answer_format": "tests"},
        {"subset_splits": {"math": "train"}},
        {"fields": {"answer": "answer"}},
        {"fields": {"prompt": "problem"}},
        {"fields": {"prompt": "problem", "answer": "answer", "typo": "column"}},
        {"fields": {"prompt": "", "answer": "answer"}},
        {"filters": {"unknown": True}},
        {"filters": {"answer_nonempty": "true"}},
        {"commit_text": "false"},
        {"dedupe": "false"},
        {"prompt_transform": "unknown"},
        {"prompt_transform": []},
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


def test_unknown_transform_rejected_at_spec_load(tmp_path: Path) -> None:
    original = SOURCES.path("aime_2025").read_text()
    (tmp_path / "aime_2025.yaml").write_text(original + "prompt_transform: typo\n")
    with pytest.raises(ValueError, match="unknown prompt_transform"):
        Registry("tasks", tmp_path, TaskSourceSpec).load("aime_2025")
