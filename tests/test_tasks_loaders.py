"""Synthetic local rows exercise the exact registry schemas without HF access."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from marli.tasks.loaders import load_tasks, strip_dapo_wrapper
from marli.tasks.source import SOURCES
from marli.tasks.taskset import TaskSet, read_tasks, write_tasks

FIXTURE = Path(__file__).parent / "fixtures" / "tasks" / "sources.json"


def local_loader(hf_id: str, config: str | None, *, split: str) -> list[dict[str, Any]]:
    source = next(spec for spec in SOURCES.load_all().values() if spec.hf_id == hf_id)
    return json.loads(FIXTURE.read_text())[source.name]


@pytest.mark.parametrize("name", SOURCES.names())
def test_all_source_schemas(name: str) -> None:
    source = SOURCES.load(name)
    rows = json.loads(FIXTURE.read_text())[name]
    tasks = load_tasks(source, loader=local_loader)
    assert tasks
    task = tasks[0]
    assert task.task_id.startswith(f"{name}/")
    assert task.prompt.startswith("Synthetic:")
    assert isinstance(task.answer, str)
    assert task.meta["source"] == name
    assert task.meta["split"] == source.split
    assert task.meta["answer_format"] == source.answer_format
    assert task.meta["row_index"] == 0
    assert set(task.meta) == {
        "source",
        "split",
        "answer_format",
        "difficulty",
        "topic",
        "row_index",
    }
    if name.startswith(("aime_", "hmmt_")):
        assert task.task_id == f"{name}/1"
        assert task.answer == str(rows[0]["answer"])
    elif name == "imo_answerbench":
        assert task.task_id == f"{name}/synthetic-1"
        assert task.answer == rows[0]["Short Answer"]
    elif name == "math500":
        assert task.task_id == f"{name}/synthetic/algebra/1"
        assert task.meta["difficulty"] == 2.0
        assert task.meta["topic"] == "Algebra"
    elif name == "deepmath_103k":
        assert task.task_id == f"{name}/0"
        assert task.answer == "9"
        assert task.meta["difficulty"] == 4.25
        assert task.meta["topic"] == "Geometry"
        assert "Ignored" not in str(task)


def test_default_backend_is_lazy_and_passes_config_and_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_loader(hf_id: str, config: str | None, *, split: str) -> list[dict[str, Any]]:
        calls.append((hf_id, config, split))
        return [{"problem_idx": 4, "problem": "Synthetic: 1 + 1?", "answer": 2}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_loader))
    source = replace(SOURCES.load("aime_2025"), config="synthetic-subset")
    tasks = load_tasks(source, split="validation")
    assert calls == [(source.hf_id, "synthetic-subset", "validation")]
    assert tasks[0].meta["split"] == "validation"


def test_filter_empty_answers_keeps_zero_and_original_row_identity() -> None:
    tasks = load_tasks(SOURCES.load("hmmt_feb_2026"), loader=local_loader)
    assert [task.answer for task in tasks] == [r"\frac{7}{4}", "0"]
    assert [task.task_id for task in tasks] == ["hmmt_feb_2026/1", "hmmt_feb_2026/5"]
    assert [task.meta["row_index"] for task in tasks] == [0, 4]


def test_difficulty_fractions_and_floats() -> None:
    tasks = load_tasks(SOURCES.load("polaris_53k"), loader=local_loader)
    assert [task.meta["difficulty"] for task in tasks] == [0.875, 0.5]
    assert [task.task_id for task in tasks] == ["polaris_53k/0", "polaris_53k/1"]


def test_nested_paths_deduplication_and_preserved_prompt() -> None:
    tasks = load_tasks(SOURCES.load("dapo_math_17k"), loader=local_loader)
    assert [task.task_id for task in tasks] == ["dapo_math_17k/101", "dapo_math_17k/103"]
    assert [task.meta["row_index"] for task in tasks] == [0, 2]
    assert [task.answer for task in tasks] == ["13", "14"]
    assert tasks[0].prompt == "Synthetic: compute 11 + 2."
    assert (
        tasks[1].prompt
        == "Synthetic: compute 11 + 3.\n  Preserve this indentation.\n\nKeep this paragraph."
    )


def test_seeded_shuffle_then_truncate_without_touching_global_rng() -> None:
    source = SOURCES.load("beyond_aime")
    full = load_tasks(source, loader=local_loader)
    expected = full.copy()
    random.Random(37).shuffle(expected)
    state = random.getstate()
    sampled = load_tasks(source, seed=37, shuffle=True, max_n=2, loader=local_loader)
    assert random.getstate() == state
    assert sampled == expected[:2]
    assert sampled == load_tasks(source, seed=37, shuffle=True, max_n=2, loader=local_loader)
    assert load_tasks(source, max_n=2, loader=local_loader) == full[:2]
    assert load_tasks(source, max_n=0, loader=local_loader) == []


def test_deduplication_precedes_truncation() -> None:
    tasks = load_tasks(SOURCES.load("dapo_math_17k"), max_n=2, loader=local_loader)
    assert len(tasks) == 2
    assert tasks[1].meta["row_index"] == 2


def test_deduplication_keeps_first_row_before_filtering() -> None:
    def loader(hf_id: str, config: str | None, *, split: str) -> list[dict[str, Any]]:
        rows = local_loader(hf_id, config, split=split)
        rows[-1]["problem"] = rows[1]["problem"]
        return rows

    source = replace(SOURCES.load("hmmt_feb_2026"), dedupe=True)
    tasks = load_tasks(source, loader=loader)
    assert [task.task_id for task in tasks] == ["hmmt_feb_2026/1"]


def test_prompt_transform_follows_spec_not_source_name() -> None:
    source = SOURCES.load("dapo_math_17k")
    renamed = load_tasks(replace(source, name="synthetic"), loader=local_loader)
    assert renamed[0].prompt == "Synthetic: compute 11 + 2."
    untransformed = load_tasks(replace(source, prompt_transform=None), loader=local_loader)
    assert untransformed[0].prompt.startswith("Solve the following math problem step by step.")


@pytest.mark.parametrize("max_n", [None, 1, 0])
def test_conflicts_drop_entire_transformed_prompt_group_and_record_manifest_counts(
    tmp_path: Path, max_n: int | None
) -> None:
    problem = "Synthetic: conflict."
    wrapped = (
        "Solve the following math problem step by step.\n\n"
        f"{problem}\nRemember to put your answer in a box."
    )
    rows = [
        (wrapped, "1"),
        ("Synthetic: agreeing.", "3"),
        (problem, "1"),
        ("Synthetic: agreeing.", 3),
        ("Synthetic: single.", "4"),
        (problem, "2"),
        (problem, "1"),
    ]

    def loader(hf_id: str, config: str | None, *, split: str) -> list[dict[str, Any]]:
        return [
            {
                "prompt": [{"content": prompt}],
                "reward_model": {"ground_truth": answer},
                "extra_info": {"index": i},
            }
            for i, (prompt, answer) in enumerate(rows)
        ]

    source = SOURCES.load("dapo_math_17k")
    meta: dict[str, Any] = {"fixture": True}
    tasks = load_tasks(source, max_n=max_n, loader=loader, meta=meta)
    expected = ["dapo_math_17k/1", "dapo_math_17k/4"][:max_n]
    assert [task.task_id for task in tasks] == expected
    assert meta == {"fixture": True, "n_raw": 7, "n_duplicates": 4, "n_conflicting_dropped": 1}
    write_tasks(tmp_path, tasks)
    handle = TaskSet(
        root=tmp_path,
        tasks="tasks.jsonl",
        source=source.name,
        split=source.split,
        kind=source.kind,
        n=len(tasks),
        answer_format=source.answer_format,
        commit_text=source.commit_text,
        meta=meta,
    )
    manifest = handle.save()
    saved = TaskSet.load(manifest)
    assert read_tasks(saved) == tasks
    assert json.loads(manifest.read_text())["meta"] == {**meta, "task_kind": "math"}


def test_no_deduplication_retains_conflicts_and_counts_raw_rows() -> None:
    def loader(hf_id: str, config: str | None, *, split: str) -> list[dict[str, Any]]:
        return [{"problem": "Synthetic", "answer": "1"}, {"problem": "Synthetic", "answer": "2"}]

    meta: dict[str, Any] = {}
    tasks = load_tasks(SOURCES.load("beyond_aime"), loader=loader, meta=meta)
    assert [task.answer for task in tasks] == ["1", "2"]
    assert meta == {"n_raw": 2, "n_duplicates": 0, "n_conflicting_dropped": 0}


@pytest.mark.parametrize("max_n", [-1, 1.5, True])
def test_bad_limit_rejected_before_loading(max_n: Any) -> None:
    with pytest.raises(ValueError, match="max_n"):
        load_tasks(SOURCES.load("aime_2025"), max_n=max_n, loader=local_loader)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_strip_dapo_wrapper_preserves_problem_bytes(newline: str) -> None:
    problem = f"  Synthetic statement.  {newline}{newline}  A second paragraph."
    wrapped = (
        "Solve the following math problem step by step. The last line contains the answer."
        f"{newline}{newline}{problem}{newline}{newline}"
        f"Remember to put your answer in a box.{newline}"
    )
    assert strip_dapo_wrapper(wrapped) == problem


@pytest.mark.parametrize(
    "prompt",
    [
        "  Synthetic unwrapped problem.\n\nWith a paragraph.\n",
        "Synthetic: Remember to put your answer inside this sentence.",
        "Solve the following math problem step by step. No blank line.",
        "Synthetic.\nRemember to put your answer in a box.\nMore problem text.",
    ],
)
def test_strip_dapo_wrapper_does_not_strip_problem_content(prompt: str) -> None:
    assert strip_dapo_wrapper(prompt) == prompt
