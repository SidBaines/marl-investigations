"""Preserve code-test semantics without trusting dataset pickle or exposing hidden cases.

Functional inputs are newline-separated JSON arguments; outputs are one JSON
value. ``tests`` includes every case, while ``public`` contains only explicitly
published examples. DeepCoder has no public-test column, so its public list is
empty unless one is explicitly supplied; examples in the problem stay intact.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import pickle
import pickletools
import unicodedata
import zlib
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from marli.envs.base import Task
from marli.tasks.source import TaskSourceSpec

if TYPE_CHECKING:
    from marli.tasks.loaders import DatasetLoader

logger = logging.getLogger(__name__)


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        allowed = {
            "str": str,
            "list": list,
            "dict": dict,
            "int": int,
            "float": float,
            "bool": bool,
            "NoneType": type(None),
        }
        if module == "builtins" and name in allowed:
            return allowed[name]
        raise pickle.UnpicklingError("private tests contain a forbidden pickle global")


def _check_pickle_value(value: Any) -> None:
    # Primitive pickle opcodes can construct types without calling find_class.
    if type(value) in (str, int, float, bool, type(None)):
        return
    if type(value) is list:
        for item in value:
            _check_pickle_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            _check_pickle_value(key)
            _check_pickle_value(item)
        return
    raise pickle.UnpicklingError("private tests contain a forbidden pickle value")


def decode_private_tests(payload: str) -> list[dict[str, Any]]:
    """Decode LCB's base64/zlib/pickle/JSON envelope without importing pickle globals."""
    try:
        raw = zlib.decompress(base64.b64decode(payload, validate=True))
        # Cached extension globals bypass Unpickler.find_class entirely.
        if any(op.name in ("EXT1", "EXT2", "EXT4") for op, _, _ in pickletools.genops(raw)):
            raise pickle.UnpicklingError("pickle extension globals are forbidden")
        value = _RestrictedUnpickler(io.BytesIO(raw)).load()
        _check_pickle_value(value)
        if not isinstance(value, str):
            raise ValueError("private test pickle must contain a JSON string")
        tests = json.loads(value)
        if not isinstance(tests, list):
            raise ValueError("private tests must be a list")
        return tests
    except (
        ValueError,
        TypeError,
        zlib.error,
        pickle.UnpicklingError,
        EOFError,
        RecursionError,
        OverflowError,
    ):
        # Dataset bytes and exception arguments can contain benchmark text.
        raise ValueError("invalid or unsafe LCB private-test payload") from None


def _metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    return value


def _cases(raw: Any) -> list[dict[str, Any]]:
    if raw is None or raw == "":
        return []
    raw = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(raw, dict) and "inputs" in raw and "outputs" in raw:
        inputs, outputs = raw["inputs"], raw["outputs"]
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise ValueError("TACO inputs and outputs must be arrays")
        if len(inputs) != len(outputs):
            raise ValueError("TACO inputs and outputs must align")
        fn_name = raw.get("fn_name")
        cases = []
        for inp, out in zip(inputs, outputs, strict=True):
            if fn_name:
                if not isinstance(inp, list) or not isinstance(out, list) or len(out) != 1:
                    raise ValueError("TACO functional cases need arguments and one result")
                inp = "\n".join(json.dumps(arg) for arg in inp)
                out = json.dumps(out[0])
            elif isinstance(out, list) and len(out) == 1:
                out = out[0]
            cases.append(
                {
                    "input": inp,
                    "output": out,
                    "testtype": "functional" if fn_name else "stdin",
                    "metadata": {"func_name": fn_name},
                }
            )
        return cases
    if not isinstance(raw, list):
        raise ValueError("tests must be a list or TACO input/output arrays")
    return raw


def _normalize_cases(
    cases: list[dict[str, Any]], metadata: dict[str, Any]
) -> tuple[str | None, str | None, list[dict[str, str]]]:
    kind: str | None = None
    fn_name: str | None = None
    normalized = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each test must be an object")
        test_meta = _metadata(case.get("metadata"))
        name = test_meta.get("func_name") or metadata.get("func_name")
        test_type = case.get("testtype") or case.get("type") or ("functional" if name else "stdin")
        if test_type not in ("stdin", "stdin_stdout", "functional"):
            raise ValueError("unknown code test type")
        case_kind = "functional" if test_type == "functional" else "stdin"
        if case_kind == "functional":
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("functional tests require a function name")
        else:
            name = None
        if kind is not None and (kind, fn_name) != (case_kind, name):
            raise ValueError("a task cannot mix test interfaces or function names")
        kind, fn_name = case_kind, name
        inp, out = case["input"], case["output"]
        if not isinstance(inp, str) or not isinstance(out, str):
            raise ValueError("test inputs and outputs must be strings")
        if kind == "functional":
            for line in inp.splitlines():
                json.loads(line)
            json.loads(out)
        normalized.append({"input": inp, "output": out})
    return kind, fn_name, normalized


def normalize_code_row(
    source: TaskSourceSpec,
    row: Mapping[str, Any],
    *,
    row_index: int,
    split: str,
    subset: str | None = None,
) -> Task:
    """Normalize one row; malformed or empty tests raise without logging source text."""
    from marli.tasks.loaders import PROMPT_TRANSFORMS, _at_path

    metadata = _metadata(row.get("metadata"))
    public = _cases(row.get("public_test_cases", []))
    if source.test_format == "lcb":
        hidden = decode_private_tests(row["private_test_cases"])
        cases = public + hidden
    else:
        cases = _cases(row.get("tests"))
    kind, fn_name, tests = _normalize_cases(cases, metadata)
    public_kind, public_fn, examples = _normalize_cases(public, metadata)
    if examples and (public_kind, public_fn) != (kind, fn_name):
        raise ValueError("public and hidden interfaces must agree")
    if not tests:
        raise ValueError("no tests")
    prompt = _at_path(row, source.prompt_path or source.fields["prompt"])
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("code prompt must be nonempty text")
    if source.prompt_transform is not None:
        prompt = PROMPT_TRANSFORMS[source.prompt_transform](prompt)
    starter = row.get("starter_code") or ""
    if not isinstance(starter, str):
        raise ValueError("starter_code must be text")
    task_id = _at_path(row, source.fields["id"]) if "id" in source.fields else row_index
    identity = f"{subset}/{task_id}" if subset is not None else str(task_id)
    return Task(
        task_id=f"{source.name}/{identity}",
        prompt=prompt,
        answer={"kind": kind, "fn_name": fn_name, "tests": tests, "public": examples},
        meta={
            "source": source.name,
            "split": split,
            "subset": subset,
            "row_index": row_index,
            "answer_format": "tests",
            "platform": row.get("platform", metadata.get("platform", subset)),
            "difficulty": (
                _at_path(row, source.fields["difficulty"])
                if "difficulty" in source.fields
                else row.get("difficulty", metadata.get("difficulty"))
            ),
            "contest_date": row.get("contest_date", metadata.get("contest_date")),
            "starter_code": starter,
            "n_tests": len(tests),
        },
    )


def _source_rows(
    source: TaskSourceSpec, subset: str | None, split: str, loader: DatasetLoader | None
) -> Iterable[Mapping[str, Any]]:
    if loader is not None:
        kwargs = {"data_files": source.data_files} if source.data_files is not None else {}
        yield from loader(source.hf_id, subset, split=split, **kwargs)
    elif source.test_format == "lcb":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(source.hf_id, source.data_files, repo_type="dataset")
        with Path(path).open() as stream:
            for line in stream:
                yield json.loads(line)
    else:
        from datasets import load_dataset

        yield from load_dataset(source.hf_id, subset, split=split, streaming=True)


def load_code_tasks(
    source: TaskSourceSpec,
    *,
    split: str | None = None,
    loader: DatasetLoader | None = None,
    meta: dict[str, Any] | None = None,
) -> list[Task]:
    """Filter inclusive contest dates, drop unusable rows, then dedupe normalized text.

    Counts describe the full pool before the caller shuffles or truncates.
    Dedupe keeps the first valid row, without merging different hidden suites.
    """
    selected_split = source.split if split is None else split
    counts = dict(n_raw=0, n_duplicates=0, n_no_tests=0, n_unparseable=0, n_filtered=0)
    tasks: list[Task] = []
    seen: set[str] = set()
    for subset in source.subsets or [source.config]:
        for row_index, row in enumerate(_source_rows(source, subset, selected_split, loader)):
            counts["n_raw"] += 1
            try:
                if source.filters:
                    contest = date.fromisoformat(row["contest_date"][:10]).isoformat()
                    if not (
                        source.filters.get("contest_date_min", "")
                        <= contest
                        <= source.filters.get("contest_date_max", "9999-12-31")
                    ):
                        counts["n_filtered"] += 1
                        continue
                task = normalize_code_row(
                    source, row, row_index=row_index, split=selected_split, subset=subset
                )
            except (ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
                counts["n_no_tests" if str(exc) == "no tests" else "n_unparseable"] += 1
                continue
            key = " ".join(unicodedata.normalize("NFKC", task.prompt).split())
            if source.dedupe and key in seen:
                counts["n_duplicates"] += 1
                continue
            seen.add(key)
            tasks.append(task)
    if meta is not None:
        meta.update(counts)
    logger.info("Code source %s: %s; kept=%d", source.name, counts, len(tasks))
    return tasks
