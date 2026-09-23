"""Use the same math judgement for rewards and votes, with killable symbolic checks.

SymPy may hang in native code and math-verify installs main-thread signal handlers.
Each symbolic check therefore runs in a fresh-interpreter process pool with a hard
per-task timeout; no symbolic work runs on the event loop or in a thread.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import re
from contextlib import suppress
from math import isfinite
from types import TracebackType
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pebble import ProcessPool


def _closing_brace(text: str, opening: int) -> int | None:
    depth = 0
    index = opening
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def extract_boxed(text: str) -> str | None:
    """Extract the last balanced \\boxed{...} or \\fbox{...}, including nested braces."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(matches):
        opening = match.end() - 1
        closing = _closing_brace(text, opening)
        if closing is not None:
            return text[opening + 1 : closing]
    return None


def normalize_answer(s: str) -> str:
    r"""Remove presentation syntax, without performing symbolic simplification.

    Select the last boxed answer if present, remove whitespace, outer math
    delimiters ($, $$, \(...\), \[...\]), a trailing period, \text{} wrappers,
    and \left/\right sizing commands. Strip commas only from a plain integer
    with groups of three digits, so tuple and decimal commas remain meaningful.
    """
    boxed = extract_boxed(s)
    answer = s if boxed is None else boxed
    answer = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", answer)
    while match := re.search(r"\\text\s*\{", answer):
        opening = match.end() - 1
        closing = _closing_brace(answer, opening)
        if closing is None:
            break
        answer = answer[: match.start()] + answer[opening + 1 : closing] + answer[closing + 1 :]
    answer = re.sub(r"\s+", "", answer)
    while True:
        previous = answer
        answer = answer.removesuffix(".")
        for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if (
                len(answer) >= len(left) + len(right)
                and answer.startswith(left)
                and answer.endswith(right)
            ):
                answer = answer[len(left) : -len(right)]
                break
        if answer == previous:
            break
    if re.fullmatch(r"[+-]?[0-9]{1,3}(?:,[0-9]{3})+", answer):
        answer = answer.replace(",", "")
    return answer


def _integer_key(answer: str) -> str | None:
    if re.fullmatch(r"[+-]?[0-9]+", answer):
        # Canonicalize arbitrary-length integers without Python's int string limit.
        digits = answer.lstrip("+-").lstrip("0") or "0"
        return ("-" if answer.startswith("-") and digits != "0" else "") + digits
    return None


def _initialize_worker() -> None:
    # Cold imports must finish before Pebble starts a check's execution timeout.
    # Defer import failures to the scheduled check's future; raising in an
    # initializer would make Pebble respawn workers indefinitely instead.
    with suppress(Exception):
        import math_verify  # noqa: F401


def _verify_symbolic(pred: str, gold: str) -> bool:
    from math_verify import parse, verify

    for boxed in (False, True):
        gold_wrapped = f"$\\boxed{{{gold}}}$" if boxed else f"${gold}$"
        pred_wrapped = f"$\\boxed{{{pred}}}$" if boxed else f"${pred}$"
        if verify(parse(gold_wrapped), parse(pred_wrapped)):
            return True
    return False


class MathVerifier:
    """Async comparisons with a lazy process pool; close after the last use.

    ``timeout_s`` bounds worker execution, not time waiting behind queued checks
    or interpreter startup. Stats count failed scheduled checks, not wrong answers.
    """

    def __init__(self, max_workers: int = 4, timeout_s: float = 5.0) -> None:
        if type(max_workers) is not int or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        if not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.max_workers = max_workers
        self.timeout_s = timeout_s
        self.stats: dict[str, int] = {"timeouts": 0, "errors": 0}
        self._pool: ProcessPool | None = None
        self._closed = False

    async def verify(self, pred: str | None, gold: str, answer_format: str) -> bool:
        if self._closed:
            raise RuntimeError("MathVerifier is closed")
        if answer_format not in ("integer", "latex"):
            raise ValueError("answer_format must be 'integer' or 'latex'")
        if pred is None:
            return False
        predicted, expected = normalize_answer(pred), normalize_answer(gold)
        if not predicted or not expected:
            return False
        if answer_format == "integer":
            pred_key, gold_key = _integer_key(predicted), _integer_key(expected)
            if pred_key is not None and gold_key is not None:
                return pred_key == gold_key
        if self._pool is None:
            from pebble import ProcessPool

            self._pool = ProcessPool(
                max_workers=self.max_workers,
                context=multiprocessing.get_context("forkserver"),
                initializer=_initialize_worker,
            )
        try:
            future = self._pool.schedule(
                _verify_symbolic, args=(predicted, expected), timeout=self.timeout_s
            )
            return bool(await asyncio.wrap_future(future))
        except TimeoutError:
            self.stats["timeouts"] += 1
        except Exception:
            # Worker exceptions and crashes are failed checks; cancellation still propagates.
            self.stats["errors"] += 1
        return False

    async def canonical(self, pred: str | None, answer_format: str) -> str | None:
        if answer_format not in ("integer", "latex"):
            raise ValueError("answer_format must be 'integer' or 'latex'")
        if pred is None:
            return None
        answer = normalize_answer(pred)
        if not answer:
            return None
        if answer_format == "integer":
            return _integer_key(answer) or answer
        return answer

    def close(self) -> None:
        """Finish queued checks and join the pool; repeated closes are harmless."""
        self._closed = True
        if self._pool is not None:
            pool, self._pool = self._pool, None
            pool.close()
            pool.join()

    async def __aenter__(self) -> Self:
        if self._closed:
            raise RuntimeError("MathVerifier is closed")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await asyncio.to_thread(self.close)
