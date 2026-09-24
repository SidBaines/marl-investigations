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
import os
import pickle
import pickletools
import random
import unicodedata
import zlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from marli.envs.base import Task
from marli.tasks.source import TaskSourceSpec

if TYPE_CHECKING:
    from marli.tasks.loaders import DatasetLoader

logger = logging.getLogger(__name__)
_MAX_PRIVATE_TEST_BYTES = 512 * 1024**2


class _CodeRowError(ValueError):
    """A fixed, benchmark-text-free reason safe to include in manifest counts."""


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
    # Memoized containers may form DAGs or cycles; visit each object only once.
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if type(item) in (str, int, float, bool, type(None)):
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        if type(item) is list:
            pending.extend(item)
        elif type(item) is dict:
            pending.extend(item.keys())
            pending.extend(item.values())
        else:
            raise pickle.UnpicklingError("private tests contain a forbidden pickle value")


def decode_private_tests(payload: str) -> list[dict[str, Any]]:
    """Decode LCB's base64/zlib/pickle/JSON envelope without importing pickle globals."""
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(
            base64.b64decode(payload, validate=True), _MAX_PRIVATE_TEST_BYTES + 1
        )
        if len(raw) > _MAX_PRIVATE_TEST_BYTES or not decoder.eof:
            raise ValueError("private tests exceed the decompression limit or are truncated")
        # Cached extension globals bypass Unpickler.find_class entirely.
        if any(op.name in ("EXT1", "EXT2", "EXT4") for op, _, _ in pickletools.genops(raw)):
            raise pickle.UnpicklingError("pickle extension globals are forbidden")
        value = _RestrictedUnpickler(io.BytesIO(raw)).load()
        _check_pickle_value(value)
        if type(value) is not str:
            raise ValueError("private test pickle must contain a JSON string")
        tests = json.loads(value)
        if not isinstance(tests, list):
            raise ValueError("private tests must be a list")
        return tests
    except Exception:
        # Malformed pickle opcodes can also raise AttributeError (BUILD/APPEND),
        # among other exceptions. Never leak dataset bytes through diagnostics.
        raise _CodeRowError("invalid or unsafe LCB private-test payload") from None


def _metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise _CodeRowError("metadata must be an object")
    return value


def _cases(raw: Any) -> list[dict[str, Any]]:
    if raw is None or raw == "":
        return []
    raw = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(raw, dict) and "inputs" in raw and "outputs" in raw:
        inputs, outputs = raw["inputs"], raw["outputs"]
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise _CodeRowError("TACO inputs and outputs must be arrays")
        if len(inputs) != len(outputs):
            raise _CodeRowError("TACO inputs and outputs must align")
        fn_name = raw.get("fn_name")
        cases = []
        for inp, out in zip(inputs, outputs, strict=True):
            if fn_name:
                if not isinstance(inp, list) or not isinstance(out, list) or len(out) != 1:
                    raise _CodeRowError("TACO functional cases need arguments and one result")
                inp = "\n".join(json.dumps(arg) for arg in inp)
                out = json.dumps(out[0])
            else:
                # TACO/APPS stdin cases store lines, not function arguments.
                if isinstance(inp, list):
                    inp = "\n".join(inp)
                if isinstance(out, list):
                    out = "\n".join(out)
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
        raise _CodeRowError("tests must be a list or TACO input/output arrays")
    return raw


def _normalize_cases(
    cases: list[dict[str, Any]], metadata: dict[str, Any]
) -> tuple[str | None, str | None, list[dict[str, str]]]:
    kind: str | None = None
    fn_name: str | None = None
    normalized = []
    for case in cases:
        if not isinstance(case, dict):
            raise _CodeRowError("each test must be an object")
        test_meta = _metadata(case.get("metadata"))
        name = case.get("fn_name") or test_meta.get("func_name") or metadata.get("func_name")
        test_type = case.get("testtype") or case.get("type") or ("functional" if name else "stdin")
        if test_type not in ("stdin", "stdin_stdout", "functional", "function_call"):
            raise _CodeRowError("unknown code test type")
        case_kind = "functional" if test_type in ("functional", "function_call") else "stdin"
        if case_kind == "functional":
            if not isinstance(name, str) or not name.isidentifier():
                raise _CodeRowError("functional tests require a function name")
        else:
            name = None
        if kind is not None and (kind, fn_name) != (case_kind, name):
            raise _CodeRowError("a task cannot mix test interfaces or function names")
        kind, fn_name = case_kind, name
        inp, out = case["input"], case["output"]
        if test_type == "function_call":
            if not isinstance(inp, list) or not isinstance(out, list) or len(out) != 1:
                raise _CodeRowError("function_call needs arguments and one result")
            inp = "\n".join(json.dumps(arg) for arg in inp)
            out = json.dumps(out[0])
        if not isinstance(inp, str) or not isinstance(out, str):
            raise _CodeRowError("test inputs and outputs must be strings")
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
        cases = public + _cases(row.get("tests"))
    kind, fn_name, tests = _normalize_cases(cases, metadata)
    public_kind, public_fn, examples = _normalize_cases(public, metadata)
    if examples and (public_kind, public_fn) != (kind, fn_name):
        raise _CodeRowError("public and hidden interfaces must agree")
    if not tests:
        raise _CodeRowError("no tests")
    prompt = _at_path(row, source.prompt_path or source.fields["prompt"])
    if not isinstance(prompt, str) or not prompt.strip():
        raise _CodeRowError("code prompt must be nonempty text")
    if source.prompt_transform is not None:
        prompt = PROMPT_TRANSFORMS[source.prompt_transform](prompt)
    starter = row.get("starter_code") or ""
    if not isinstance(starter, str):
        raise _CodeRowError("starter_code must be text")
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

        path = hf_hub_download(
            source.hf_id,
            source.data_files,
            repo_type="dataset",
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "").upper()
            in {"1", "ON", "YES", "TRUE"},
        )
        with Path(path).open() as stream:
            for line in stream:
                yield json.loads(line)
    else:
        from huggingface_hub import snapshot_download
        from pyarrow.parquet import ParquetFile

        pattern = f"{subset}/{split}-*.parquet"
        root = snapshot_download(
            source.hf_id,
            repo_type="dataset",
            allow_patterns=[pattern],
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "").upper()
            in {"1", "ON", "YES", "TRUE"},
        )
        paths = sorted(Path(root).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no parquet shards for {source.name}/{subset}/{split}")
        columns = {
            path.split(".")[0] for path in [*source.fields.values(), source.prompt_path or ""]
        } | {
            "tests",
            "public_test_cases",
            "metadata",
            "starter_code",
            "platform",
            "difficulty",
            "contest_date",
        }
        for path in paths:
            with ParquetFile(path) as parquet:
                selected = sorted(columns.intersection(parquet.schema_arrow.names))
                # A single test bundle can be hundreds of MiB. Never prefetch a
                # large batch, or read reference solutions we do not use.
                for batch in parquet.iter_batches(
                    batch_size=1, columns=selected, use_threads=False
                ):
                    yield from batch.to_pylist()


def load_code_tasks(
    source: TaskSourceSpec,
    *,
    split: str | None = None,
    loader: DatasetLoader | None = None,
    meta: dict[str, Any] | None = None,
    seed: int = 0,
) -> Iterator[Task]:
    """Filter inclusive contest dates, drop unusable rows, then dedupe normalized text.

    Counts describe the full pool on exhaustion, before shuffle or truncation.
    Only one row's tests are resident; dedupe retains normalized prompt keys.
    Caps apply to hidden tests after dedupe, first by UTF-8 bytes then a seeded
    sample keyed by task identity (independent of iteration/shuffle order).
    Dedupe keeps the first valid row, without merging different hidden suites.
    Per-subset splits override the source default; an explicit split overrides both.
    """
    counts = dict(n_raw=0, n_duplicates=0, n_no_tests=0, n_unparseable=0, n_filtered=0)
    test_counts = dict(n_tests_dropped_bytes=0, n_tests_dropped_cap=0)
    reasons: dict[str, int] = {}
    kept = 0
    seen: set[str] = set()
    for subset in source.subsets or [source.config]:
        selected_split = source.subset_splits.get(subset, source.split) if split is None else split
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
                        reasons["contest_date"] = reasons.get("contest_date", 0) + 1
                        continue
                task = normalize_code_row(
                    source, row, row_index=row_index, split=selected_split, subset=subset
                )
                key = " ".join(unicodedata.normalize("NFKC", task.prompt).split())
                if source.dedupe and key in seen:
                    counts["n_duplicates"] += 1
                    reasons["duplicate"] = reasons.get("duplicate", 0) + 1
                    continue
                public = task.answer["public"]
                hidden = task.answer["tests"][len(public) :]
                if source.max_test_bytes is not None:
                    small = [
                        case
                        for case in hidden
                        if len(case["input"].encode("utf-8")) + len(case["output"].encode("utf-8"))
                        <= source.max_test_bytes
                    ]
                    test_counts["n_tests_dropped_bytes"] += len(hidden) - len(small)
                    hidden = small
                if source.max_tests is not None and len(hidden) > source.max_tests:
                    indices = sorted(
                        random.Random(f"{seed}:{task.task_id}").sample(
                            range(len(hidden)), source.max_tests
                        )
                    )
                    test_counts["n_tests_dropped_cap"] += len(hidden) - len(indices)
                    hidden = [hidden[index] for index in indices]
                tests = public + hidden
                if not tests:
                    raise _CodeRowError("no tests after caps")
            except (ValueError, TypeError, KeyError, IndexError, RecursionError) as exc:
                reason = str(exc) if isinstance(exc, _CodeRowError) else type(exc).__name__
                counts[
                    "n_no_tests"
                    if reason in ("no tests", "no tests after caps")
                    else "n_unparseable"
                ] += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            seen.add(key)
            kept += 1
            yield replace(
                task,
                answer={**task.answer, "tests": tests},
                meta={**task.meta, "n_tests": len(tests)},
            )
    if meta is not None:
        meta.update(
            **counts,
            **test_counts,
            drop_reasons=reasons,
            max_tests=source.max_tests,
            max_test_bytes=source.max_test_bytes,
        )
    logger.info("Code source %s: %s; kept=%d", source.name, counts, kept)
