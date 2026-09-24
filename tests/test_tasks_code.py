"""Test code-source semantics with synthetic bytes, plus read-only cache smoke checks."""

from __future__ import annotations

import base64
import copyreg
import json
import os
import random
import subprocess
import sys
import zlib
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _marli_code_fixtures import deepcoder_row, lcb_row, private_payload

from marli.cli.main import main
from marli.data import build as build_module
from marli.data.build import BuildConfig
from marli.handles import InputRef
from marli.tasks.code import decode_private_tests, normalize_code_row
from marli.tasks.loaders import load_tasks
from marli.tasks.source import SOURCES
from marli.tasks.taskset import TaskSet, read_tasks
from marli.verbs import run_verb


def test_code_registry_specs() -> None:
    deepcoder, lcb = SOURCES.load("deepcoder"), SOURCES.load("lcb_v6")
    assert deepcoder.kind == lcb.kind == "code"
    assert deepcoder.answer_format == lcb.answer_format == "tests"
    assert deepcoder.test_format == "deepcoder" and lcb.test_format == "lcb"
    assert deepcoder.subsets == ["primeintellect", "taco", "lcbv5", "codeforces"]
    assert deepcoder.split == "train" and deepcoder.dedupe
    assert deepcoder.subset_splits == {
        "primeintellect": "train",
        "taco": "train",
        "lcbv5": "train",
        "codeforces": "test",
    }
    assert lcb.data_files == "test6.jsonl"
    assert lcb.filters == {"contest_date_min": "2025-02-01", "contest_date_max": "2025-05-01"}
    assert not deepcoder.commit_text and not lcb.commit_text


@pytest.mark.parametrize("functional", [False, True])
def test_lcb_normalization(functional: bool) -> None:
    row = lcb_row(functional=functional)
    task = normalize_code_row(SOURCES.load("lcb_v6"), row, row_index=2, split="test")
    assert task.task_id == "lcb_v6/synthetic-sum"
    assert task.answer == {
        "kind": "functional" if functional else "stdin",
        "fn_name": "add" if functional else None,
        "tests": [
            {"input": "2\n3" if functional else "2 3\n", "output": "5"},
            {"input": "4\n5" if functional else "4 5\n", "output": "9"},
        ],
        "public": [{"input": "2\n3" if functional else "2 3\n", "output": "5"}],
    }
    assert task.meta == {
        "source": "lcb_v6",
        "split": "test",
        "subset": None,
        "row_index": 2,
        "answer_format": "tests",
        "platform": row["platform"],
        "difficulty": "easy",
        "contest_date": row["contest_date"],
        "starter_code": row["starter_code"],
        "n_tests": 2,
    }


@pytest.mark.parametrize("subset", ["primeintellect", "taco", "lcbv5", "codeforces"])
def test_deepcoder_subset_shapes(subset: str) -> None:
    task = normalize_code_row(
        SOURCES.load("deepcoder"),
        deepcoder_row(subset),
        row_index=7,
        split="train",
        subset=subset,
    )
    assert task.task_id == f"deepcoder/{subset}/7"
    assert task.answer["kind"] == "stdin"
    assert task.answer["fn_name"] is None and task.answer["public"] == []
    assert task.answer["tests"][0]["input"] == "2 3\n"
    assert task.meta["n_tests"] == (2 if subset == "taco" else 1)
    assert "REFERENCE_SOLUTION" not in repr(task)


@pytest.mark.parametrize("subset", ["taco", "lcbv5"])
def test_deepcoder_functional_shapes(subset: str) -> None:
    row = deepcoder_row(subset, functional=True)
    task = normalize_code_row(SOURCES.load("deepcoder"), row, row_index=0, split="train")
    assert task.answer["kind"] == "functional" and task.answer["fn_name"] == "add"
    assert task.answer["tests"][0] == {"input": "2\n3", "output": "5"}


def test_taco_preserves_list_arguments_and_list_results() -> None:
    row = deepcoder_row("taco", functional=True)
    row["tests"] = json.dumps(
        {
            "fn_name": "identity",
            "inputs": [[[1, 2]]],
            "outputs": [[[1, 2]]],
        }
    )
    task = normalize_code_row(SOURCES.load("deepcoder"), row, row_index=0, split="train")
    assert task.answer["tests"] == [{"input": "[1, 2]", "output": "[1, 2]"}]


def test_restricted_unpickler_rejects_code_execution(tmp_path: Path) -> None:
    marker = tmp_path / "unsafe"

    class Malicious:
        def __reduce__(self) -> tuple[Any, tuple[str]]:
            return os.system, (f"touch {marker}",)

    with pytest.raises(ValueError, match="unsafe"):
        decode_private_tests(private_payload(Malicious()))
    assert not marker.exists()


@pytest.mark.parametrize("value", [(), b"[]", {1, 2}, [complex(1, 2)]])
def test_unpickler_rejects_non_allowlisted_values(value: Any) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        decode_private_tests(private_payload(value))


def test_unpickler_rejects_cached_extension_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(copyreg._extension_cache, 1, eval)
    # EXT1(1), BINUNICODE("'[]'"), TUPLE1, REDUCE would invoke cached eval.
    raw = b"\x80\x02\x82\x01X\x04\x00\x00\x00'[]'\x85R."
    payload = base64.b64encode(zlib.compress(raw)).decode()
    with pytest.raises(ValueError, match="unsafe"):
        decode_private_tests(payload)


@pytest.mark.parametrize("payload", ["not base64!", private_payload("not JSON"), ""])
def test_bad_private_test_payload(payload: str) -> None:
    with pytest.raises(ValueError, match="payload"):
        decode_private_tests(payload)


def test_filter_dedupe_and_counts_precede_sampling() -> None:
    good = deepcoder_row("codeforces")
    duplicate = {**good, "problem": "  Synthetic:\n add two integers.  "}
    empty = {**good, "tests": "[]"}
    bad = {**good, "tests": "not json"}
    other = {**good, "problem": "Synthetic: another sum."}
    rows = [empty, good, duplicate, bad, other]
    calls = []

    def loader(hf_id: str, subset: str | None, *, split: str) -> list[dict[str, Any]]:
        calls.append((hf_id, subset, split))
        return rows

    source = replace(
        SOURCES.load("deepcoder"), subsets=["codeforces"], subset_splits={"codeforces": "test"}
    )
    meta: dict[str, Any] = {"existing": True}
    tasks = load_tasks(source, split="test", loader=loader, meta=meta)
    assert [task.task_id for task in tasks] == ["deepcoder/codeforces/1", "deepcoder/codeforces/4"]
    assert calls == [(source.hf_id, "codeforces", "test")]
    assert meta == {
        "existing": True,
        "n_raw": 5,
        "n_duplicates": 1,
        "n_no_tests": 1,
        "n_unparseable": 1,
        "n_filtered": 0,
    }
    state = random.getstate()
    expected = tasks.copy()
    random.Random(3).shuffle(expected)
    assert (
        load_tasks(source, loader=loader, split="test", shuffle=True, seed=3, max_n=1)
        == expected[:1]
    )
    assert random.getstate() == state
    assert load_tasks(source, loader=loader, max_n=0) == []


def test_all_subsets_and_cross_subset_deduplication() -> None:
    calls = []

    def loader(hf_id: str, subset: str, *, split: str) -> list[dict[str, Any]]:
        calls.append((subset, split))
        return [deepcoder_row(subset)]

    meta: dict[str, Any] = {}
    tasks = load_tasks(SOURCES.load("deepcoder"), loader=loader, meta=meta)
    assert calls == list(SOURCES.load("deepcoder").subset_splits.items())
    assert len(tasks) == 1 and meta["n_raw"] == 4 and meta["n_duplicates"] == 3


def test_date_window_inclusive_and_configurable() -> None:
    rows = [
        lcb_row(contest_date=day)
        for day in (
            "2025-01-31T23:59:59",
            "2025-02-01T00:00:00",
            "2025-05-01T12:00:00",
            "2025-05-02",
        )
    ]

    def loader(
        hf_id: str, subset: str | None, *, split: str, data_files: str
    ) -> list[dict[str, Any]]:
        assert data_files == "test6.jsonl" and split == "test"
        return rows

    source = SOURCES.load("lcb_v6")
    meta: dict[str, Any] = {}
    assert [task.meta["row_index"] for task in load_tasks(source, loader=loader, meta=meta)] == [
        1,
        2,
    ]
    assert meta["n_raw"] == 4 and meta["n_filtered"] == 2
    assert len(load_tasks(replace(source, filters={}), loader=loader)) == 4
    assert (
        len(load_tasks(replace(source, filters={"contest_date_min": "2025-05-02"}), loader=loader))
        == 1
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"answer_format": "integer"},
        {"answer_path": "answer"},
        {"test_format": "unknown"},
        {"test_format": None},
        {"filters": {"unknown": True}},
        {"filters": {"contest_date_min": 3}},
        {"filters": {"contest_date_min": "2025-02-30"}},
        {"filters": {"contest_date_min": "2025-05-01", "contest_date_max": "2025-02-01"}},
        {"config": "taco"},
        {"subsets": ["taco", "taco"]},
        {"subsets": [1]},
        {"subset_splits": {"typo": "train"}},
        {"subset_splits": {"taco": ""}},
        {"subset_splits": {"taco": 1}},
        {"subset_splits": []},
    ],
)
def test_code_spec_validation(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(SOURCES.load("deepcoder"), **changes)


@pytest.mark.parametrize(
    "tests",
    [
        [{"input": "x"}],
        [{"input": [], "output": "x"}],
        [{"input": "1", "output": "1", "testtype": "unknown"}],
        [{"input": "1", "output": "1", "testtype": "functional"}],
        {"inputs": ["1"], "outputs": []},
    ],
)
def test_malformed_cases_drop_entire_row(tests: Any) -> None:
    row = {**deepcoder_row("codeforces"), "tests": json.dumps(tests)}

    def loader(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [row]

    meta: dict[str, Any] = {}
    assert (
        load_tasks(
            replace(SOURCES.load("deepcoder"), subsets=[], subset_splits={}),
            loader=loader,
            meta=meta,
        )
        == []
    )
    assert meta["n_unparseable"] == 1


def test_imports_are_lazy() -> None:
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'datasets', 'pyarrow', 'huggingface_hub', 'torch', 'tinker_cookbook'}:
        raise AssertionError('eager heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import marli.tasks.code, marli.tasks.loaders, marli.envs.code_fn
from marli.tasks.source import SOURCES
SOURCES.load('deepcoder')
SOURCES.load('lcb_v6')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_lazy_deepcoder_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def loader(hf_id: str, subset: str, *, split: str, streaming: bool) -> list[dict[str, Any]]:
        assert hf_id == SOURCES.load("deepcoder").hf_id and subset == "taco"
        assert split == "train" and streaming
        return [deepcoder_row("taco")]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=loader))
    assert (
        len(
            load_tasks(
                replace(
                    SOURCES.load("deepcoder"), subsets=["taco"], subset_splits={"taco": "train"}
                )
            )
        )
        == 1
    )


def test_lcb_reads_exact_jsonl_without_dataset_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "test6.jsonl"
    path.write_text(json.dumps(lcb_row()) + "\n")

    def download(repo_id: str, filename: str, *, repo_type: str) -> str:
        assert (repo_id, filename, repo_type) == (
            SOURCES.load("lcb_v6").hf_id,
            "test6.jsonl",
            "dataset",
        )
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    assert len(load_tasks(SOURCES.load("lcb_v6"))) == 1


@pytest.mark.hf
@pytest.mark.parametrize(
    "name,subset",
    [
        ("deepcoder", "primeintellect"),
        ("deepcoder", "taco"),
        ("deepcoder", "lcbv5"),
        ("deepcoder", "codeforces"),
        ("lcb_v6", None),
    ],
)
async def test_cached_registry_rows(
    name: str, subset: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SOURCES.load(name)
    hub = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
        )
    )
    snapshots = hub / ("datasets--" + source.hf_id.replace("/", "--")) / "snapshots"
    split = "test" if subset == "codeforces" or name == "lcb_v6" else "train"
    pattern = f"*/{subset}/{split}-*.parquet" if subset else "*/test6.jsonl"
    paths = sorted(p for p in snapshots.glob(pattern) if p.is_file())
    if not paths:
        pytest.skip("HF source files are not cached")
    if subset:
        pq = pytest.importorskip("pyarrow.parquet")
        rows = next(pq.ParquetFile(paths[0]).iter_batches(batch_size=3)).to_pylist()
        source = replace(source, subsets=[subset], subset_splits={subset: split})
    else:
        rows = []
        with paths[0].open() as stream:
            for line in stream:
                row = json.loads(line)
                if (
                    source.filters["contest_date_min"]
                    <= row["contest_date"][:10]
                    <= source.filters["contest_date_max"]
                ):
                    rows.append(row)
                    if len(rows) == 3:
                        break

    def loader(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return rows

    # Use real cached bytes through the build verb without backend downloads or writes.
    monkeypatch.setattr(build_module, "load_tasks", partial(load_tasks, loader=loader))
    monkeypatch.setattr(build_module.SOURCES, "load", lambda name: source)
    result = await run_verb(
        "data build", BuildConfig(source=name, max_n=3), out=tmp_path / "cached"
    )
    handle = TaskSet.load(result.manifest)
    tasks = read_tasks(handle)
    # Assertions inspect structure only; never include source text in test output.
    assert handle.kind == "code" and handle.answer_format == "tests"
    assert len(tasks) == 3
    for task in tasks:
        assert isinstance(task.prompt, str)
        assert isinstance(task.answer, dict)
        assert task.answer["kind"] in ("stdin", "functional")
        assert task.meta["n_tests"] > 0
        assert all(isinstance(test["input"], str) for test in task.answer["tests"])


@pytest.mark.parametrize("split", [None, "validation"])
def test_subset_split_defaults_and_explicit_override(split: str | None) -> None:
    source = SOURCES.load("deepcoder")
    calls: list[tuple[str, str]] = []

    def loader(hf_id: str, subset: str, *, split: str) -> list[dict[str, Any]]:
        calls.append((subset, split))
        return [{**deepcoder_row(subset), "problem": f"Synthetic: {subset} sum."}]

    tasks = load_tasks(source, loader=loader, split=split)
    expected = [(subset, split or source.subset_splits[subset]) for subset in source.subsets]
    assert calls == expected
    assert [(task.meta["subset"], task.meta["split"]) for task in tasks] == expected
    assert len(tasks) == 4


@pytest.mark.parametrize("name", ["deepcoder", "lcb_v6"])
def test_code_data_build_cli(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def loader(hf_id: str, subset: str | None, **kwargs: Any) -> list[dict[str, Any]]:
        if subset is None:
            return [lcb_row()]
        assert kwargs["split"] == SOURCES.load("deepcoder").subset_splits[subset]
        return [{**deepcoder_row(subset), "problem": f"Synthetic: {subset} sum."}]

    monkeypatch.setattr(build_module, "load_tasks", partial(load_tasks, loader=loader))
    out = tmp_path / name
    assert main(["data", "build", f"source={name}", "--out", str(out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] and payload["kind"] == "taskset"
    handle = TaskSet.load(payload["manifest"])
    assert handle.kind == "code" and handle.answer_format == "tests"
    assert read_tasks(handle) == load_tasks(SOURCES.load(name), loader=loader)
    assert handle.n == (4 if name == "deepcoder" else 1)


async def test_deepcoder_build_excludes_normalized_lcb_text_before_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def loader(hf_id: str, subset: str | None, **kwargs: Any) -> list[dict[str, Any]]:
        if subset is None:
            return [lcb_row()]
        return [
            {**deepcoder_row(subset), "problem": "  SYNTHETIC: add\n two integers!  "},
            {**deepcoder_row(subset), "problem": f"Synthetic: {subset} unique."},
        ]

    monkeypatch.setattr(build_module, "load_tasks", partial(load_tasks, loader=loader))
    excluded = await run_verb("data build", BuildConfig(source="lcb_v6"), out=tmp_path / "eval")
    built = await run_verb(
        "data build",
        BuildConfig(source="deepcoder", exclude=str(excluded.manifest), max_n=2),
        out=tmp_path / "train",
    )
    assert [task.task_id for task in read_tasks(built.handle)] == [
        "deepcoder/primeintellect/1",
        "deepcoder/taco/1",
    ]
    assert built.handle.meta["counts"] == {
        "loaded": 5,
        "kept": 2,
        "dropped_exact": 1,
        "dropped_ngram": 0,
    }
    assert built.handle.inputs == (InputRef.of(excluded.handle),)
