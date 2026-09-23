"""Use the same math judgement for rewards and votes, with killable symbolic checks.

SymPy may hang in native code and math-verify installs main-thread signal handlers.
Each symbolic check therefore runs in a fresh-interpreter process pool with a hard
per-task timeout; no symbolic work runs on the event loop or in a thread.
Forkserver re-imports ``__main__``, so caller scripts need an
``if __name__ == "__main__":`` guard. math-verify's default ``float_rounding=6``
is retained: decimals rounded to six places can match exact irrational golds.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import re
from concurrent.futures import Future
from contextlib import suppress
from math import isfinite
from threading import Lock
from types import TracebackType
from typing import TYPE_CHECKING, Self

from marli.errors import ConfigError

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


def _light_normalize(s: str) -> str:
    """Extract an answer without changing its interior mathematical notation."""
    boxed = extract_boxed(s)
    answer = (s if boxed is None else boxed).strip()
    while True:
        for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if (
                len(answer) >= len(left) + len(right)
                and answer.startswith(left)
                and answer.endswith(right)
            ):
                answer = answer[len(left) : -len(right)].strip()
                break
        else:
            return answer


def _safe_prediction(answer: str) -> bool:
    if any(delimiter in answer for delimiter in ("$", r"\(", r"\)", r"\[", r"\]")):
        return False
    depth = 0
    # Escaped braces denote literal set braces, not TeX grouping syntax.
    for token in re.findall(r"\\.|[{}]", answer):
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def normalize_answer(s: str) -> str:
    r"""Remove presentation syntax, without performing symbolic simplification.

    Select the last boxed answer if present, remove presentation whitespace, outer math
    delimiters ($, $$, \(...\), \[...\]), a trailing period, \text{} wrappers,
    and \left/\right sizing commands. Plain integer thousands groups of exactly
    three digits may use commas, spaces, or \, separators (one style per number).
    Other whitespace between digits stays as a space: ``7 0`` is never ``70``.
    Tuple and decimal commas remain meaningful.
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
    answer = re.sub(r"\s+", " ", answer.strip())
    answer = re.sub(r"(?<![0-9]) | (?![0-9])", "", answer)
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
    for separator in (",", " ", r"\,"):
        if re.fullmatch(r"[+-]?[0-9]{1,3}(?:" + re.escape(separator) + r"[0-9]{3})+", answer):
            answer = answer.replace(separator, "")
            break
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
    with suppress(ImportError):
        import math_verify  # noqa: F401


def _verify_symbolic(pred: str, gold: str) -> bool:
    from math_verify import parse, verify

    parsed_gold = parse(f"${gold}$")
    if parsed_gold == []:
        parsed_gold = parse(f"$\\boxed{{{gold}}}$")
    parsed_pred = parse(f"${pred}$")
    if parsed_pred == []:
        parsed_pred = parse(f"$\\boxed{{{pred}}}$")
    return bool(verify(parsed_gold, parsed_pred))


class MathVerifier:
    """Async comparisons with a lazy process pool; close after the last use.

    ``timeout_s`` bounds worker execution, not time waiting behind queued checks
    or interpreter startup. Stats count timeouts, crashed workers, and rejected
    prediction syntax, not ordinary wrong answers. Other failures propagate.
    """

    def __init__(self, max_workers: int = 4, timeout_s: float = 5.0) -> None:
        if type(max_workers) is not int or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        if not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.max_workers = max_workers
        self.timeout_s = timeout_s
        self.stats: dict[str, int] = {"timeouts": 0, "errors": 0, "rejected": 0}
        self._pool: ProcessPool | None = None
        self._pool_lock = Lock()
        self._closed = False

    async def verify(self, pred: str | None, gold: str, answer_format: str) -> bool:
        if self._closed:
            raise RuntimeError("MathVerifier is closed")
        if answer_format not in ("integer", "latex"):
            raise ValueError("answer_format must be 'integer' or 'latex'")
        if pred is None:
            return False
        predicted, expected = _light_normalize(pred), _light_normalize(gold)
        if not _safe_prediction(predicted):
            self.stats["rejected"] += 1
            return False
        if not predicted or not expected:
            return False
        pred_key = _integer_key(normalize_answer(predicted))
        gold_key = _integer_key(normalize_answer(expected))
        if pred_key is not None and gold_key is not None:
            return pred_key == gold_key
        from pebble import ProcessExpired

        try:
            future = await asyncio.to_thread(self._schedule_symbolic, predicted, expected)
            return bool(await asyncio.wrap_future(future))
        except TimeoutError:
            self.stats["timeouts"] += 1
        except ProcessExpired:
            self.stats["errors"] += 1
        except ImportError as exc:
            raise ConfigError(f"math-verify unavailable: {exc}") from exc
        return False

    def _schedule_symbolic(self, pred: str, gold: str) -> Future[bool]:
        from pebble import ProcessPool

        # Pebble starts workers on the first schedule(), not in its constructor.
        # Serialize creation with close(), including cancelled startup awaits.
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("MathVerifier is closed")
            if self._pool is None:
                context = multiprocessing.get_context("forkserver")
                context.set_forkserver_preload(["math_verify"])
                self._pool = ProcessPool(
                    max_workers=self.max_workers, context=context, initializer=_initialize_worker
                )
            return self._pool.schedule(_verify_symbolic, args=(pred, gold), timeout=self.timeout_s)

    async def canonical(self, pred: str | None, answer_format: str) -> str | None:
        """Return a presentation key; this pure operation remains valid after close()."""
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
        with self._pool_lock:
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
