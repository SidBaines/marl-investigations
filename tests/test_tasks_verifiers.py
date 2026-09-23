"""One grader handles notation variants and survives hung or crashed worker processes."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import marli.tasks.verifiers as verifier_module
from marli.tasks.verifiers import MathVerifier, extract_boxed, normalize_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("no boxed answer", None),
        (r"Work: \boxed{4}. Final: \boxed{\frac{1}{2}}.", r"\frac{1}{2}"),
        (r"\boxed{3} then \fbox{7}", "7"),
        (r"\fbox {a^{b+c}}", "a^{b+c}"),
        (r"\boxed{\{1,2\}}", r"\{1,2\}"),
        (r"\boxed{4} then \boxed{unfinished", "4"),
        (r"\boxed{unfinished", None),
        (r"\boxed{}", ""),
    ],
)
def test_extract_balanced_last_box(text: str, expected: str | None) -> None:
    assert extract_boxed(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" 70 ", "70"),
        ("$70$.", "70"),
        (r"\boxed{70}", "70"),
        (r"\fbox{70}", "70"),
        (r"\text{ 70 }", "70"),
        (r"\text{\text{70}}", "70"),
        (r"\text{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\left( 3, \frac{\pi}{2} \right)", r"(3,\frac{\pi}{2})"),
        (r"\rightarrow", r"\rightarrow"),
        ("1,234,567", "1234567"),
        ("-1,234", "-1234"),
        ("12,34", "12,34"),
        ("(1,234)", "(1,234)"),
        ("1,234.5", "1,234.5"),
        (r"$$-\frac{1}{21}$$", r"-\frac{1}{21}"),
        (r"\(70\)", "70"),
        (r"\[70\]", "70"),
        ("$70.$", "70"),
    ],
)
def test_normalize_presentation_only(text: str, expected: str) -> None:
    assert normalize_answer(text) == expected


@pytest.mark.parametrize("pred", ["70", "070", " 70 ", r"\boxed{70}", "$70$", r"\text{70}", "+070"])
async def test_integer_fast_path_and_canonical(pred: str) -> None:
    async with MathVerifier() as verifier:
        assert await verifier.verify(pred, "70", "integer") is True
        assert await verifier.canonical(pred, "integer") == "70"
        assert verifier._pool is None
        assert verifier.stats == {"timeouts": 0, "errors": 0}


async def test_fast_path_handles_empty_wrong_negative_and_large_integers() -> None:
    async with MathVerifier() as verifier:
        for pred in (None, "", " ", "$$", r"\boxed{}", "71", "-70"):
            assert await verifier.verify(pred, "70", "integer") is False
        assert await verifier.verify("-001,234", "-1234", "integer")
        assert await verifier.verify("-00", "0", "integer")
        assert await verifier.verify("0" + "9" * 5000, "9" * 5000, "integer")
        assert not await verifier.verify("70", "", "integer")
        assert await verifier.canonical(None, "integer") is None
        assert await verifier.canonical(r"\boxed{}", "latex") is None
        assert await verifier.canonical(r"$\frac{1}{2}$", "latex") == r"\frac{1}{2}"
        assert verifier._pool is None


async def test_real_symbolic_variants_and_concurrent_calls() -> None:
    pytest.importorskip("pebble")
    pytest.importorskip("math_verify")
    cases = [
        (r"\frac{1}{2}", "0.5", "latex", True),
        ("0.5", r"\frac{1}{2}", "latex", True),
        (r"\left( 3, \frac{\pi}{2} \right)", r"(3, \pi/2)", "latex", True),
        (r"-\frac{1}{21}", "-1/21", "latex", True),
        (r"\boxed{-\frac{1}{21}}", "-1/21", "latex", True),
        (r"\fbox{\frac{1}{2}}", "$0.5$", "latex", True),
        (r"\(\frac{1}{2}\)", "0.5", "latex", True),
        (r"\frac{140}{2}", "70", "integer", True),
        ("70.0", "70", "integer", True),
        (r"\frac{1}{3}", "0.5", "latex", False),
        (r"\frac{1}{21}", r"-\frac{1}{21}", "latex", False),
        (r"(4, \pi/2)", r"(3, \pi/2)", "latex", False),
    ]
    async with MathVerifier(max_workers=2, timeout_s=5.0) as verifier:
        results = await asyncio.gather(
            *(verifier.verify(pred, gold, fmt) for pred, gold, fmt, _ in cases * 3)
        )
        assert results == [expected for _, _, _, expected in cases * 3]
        assert verifier.stats == {"timeouts": 0, "errors": 0}
    assert verifier._pool is None


def worker_with_hang_or_crash(pred: str, gold: str) -> bool:
    if pred == "hang":
        time.sleep(30)
    if pred == "crash":
        os._exit(7)
    return pred == gold


@pytest.mark.parametrize(("pred", "counter"), [("hang", "timeouts"), ("crash", "errors")])
async def test_worker_failure_is_bounded_and_pool_recovers(
    pred: str, counter: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pebble")
    monkeypatch.setattr(verifier_module, "_verify_symbolic", worker_with_hang_or_crash)
    async with MathVerifier(max_workers=1, timeout_s=0.2) as verifier:
        assert await verifier.verify("warm", "warm", "latex")
        start = time.monotonic()
        assert await asyncio.wait_for(verifier.verify(pred, "x", "latex"), 5.0) is False
        assert time.monotonic() - start < 5.0
        assert verifier.stats[counter] == 1
        assert sum(verifier.stats.values()) == 1
        assert await verifier.verify("recovered", "recovered", "latex")


async def test_pathological_exponent_tower_is_bounded() -> None:
    pytest.importorskip("pebble")
    pytest.importorskip("math_verify")
    tower = "2^{" * 4000 + "2" + "}" * 4000
    async with MathVerifier(max_workers=1) as verifier:
        assert await verifier.verify("x", "x", "latex")
        verifier.timeout_s = 0.2
        start = time.monotonic()
        assert await asyncio.wait_for(verifier.verify(tower, "1", "latex"), 5.0) is False
        assert time.monotonic() - start < 5.0


async def test_pool_start_method_and_hard_task_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    class Pool:
        def __init__(
            self, *, max_workers: int, context: Any, initializer: Callable[[], None]
        ) -> None:
            assert initializer is verifier_module._initialize_worker
            calls.append((max_workers, context.get_start_method()))

        def schedule(
            self, function: Callable[..., bool], *, args: tuple[str, str], timeout: float
        ) -> Future[bool]:
            calls.append((function, args, timeout))
            future: Future[bool] = Future()
            future.set_result(True)
            return future

        def close(self) -> None:
            calls.append("close")

        def join(self) -> None:
            calls.append("join")

    monkeypatch.setitem(sys.modules, "pebble", SimpleNamespace(ProcessPool=Pool))
    verifier = MathVerifier(max_workers=3, timeout_s=0.25)
    assert calls == []
    assert await verifier.verify("$x$", r"\boxed{x}", "latex")
    assert calls == [(3, "forkserver"), (verifier_module._verify_symbolic, ("x", "x"), 0.25)]
    verifier.close()
    verifier.close()
    assert calls[-2:] == ["close", "join"]
    with pytest.raises(RuntimeError, match="closed"):
        await verifier.verify("1", "1", "integer")


def test_symbolic_worker_wraps_and_retries_boxing(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = []
    checked = []

    def parse(text: str) -> str:
        parsed.append(text)
        return text

    def verify(gold: str, pred: str) -> bool:
        checked.append((gold, pred))
        return r"\boxed" in gold and r"\boxed" in pred

    monkeypatch.setitem(sys.modules, "math_verify", SimpleNamespace(parse=parse, verify=verify))
    assert verifier_module._verify_symbolic("prediction", "reference")
    assert parsed == [
        "$reference$",
        "$prediction$",
        r"$\boxed{reference}$",
        r"$\boxed{prediction}$",
    ]
    assert checked[0] == ("$reference$", "$prediction$")


def test_initialization_import_failure_is_reported_by_scheduled_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "math_verify", None)
    verifier_module._initialize_worker()
    with pytest.raises(ImportError):
        verifier_module._verify_symbolic("x", "x")


@pytest.mark.parametrize(
    "kwargs", [{"max_workers": 0}, {"timeout_s": 0}, {"timeout_s": float("inf")}]
)
def test_invalid_pool_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MathVerifier(**kwargs)


async def test_invalid_format_is_loud() -> None:
    async with MathVerifier() as verifier:
        with pytest.raises(ValueError, match="answer_format"):
            await verifier.verify("1", "1", "unknown")
        with pytest.raises(ValueError, match="answer_format"):
            await verifier.canonical("1", "unknown")


def test_task_imports_stay_light() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import marli.tasks.source
import marli.tasks.loaders
import marli.tasks.taskset
import marli.tasks.verifiers
import marli.tasks.equivalence
assert not {'datasets', 'math_verify', 'pebble', 'sympy'} & sys.modules.keys()
""",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
