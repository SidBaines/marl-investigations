"""Decision parsers for economic-game replies: explicit rules, failures are counted.

A parser returns None rather than guessing: an ambiguous or missing decision is a
parse failure, reported beside the rate it would otherwise distort. Rules run in
order; the first that yields a consistent decision wins:

1. the whole reply is the decision (``Option B``, ``**B**``, ``50``, ``50 points``);
2. lines that consist only of a decision (answer-first or answer-last replies);
3. first-person decision statements (``I choose **Option B**``, ``My minimum
   acceptable offer is 30``, ``Final answer: Option A``), ignoring conditionals
   such as ``if I choose Option A``; disagreeing statements resolve to the last;
4. options only: exactly one option mentioned anywhere;
5. emphasised (bold/boxed) decisions that all agree;
6. amounts only: one number on the last line, else one number in the whole reply.

Replies are first cut after the last ``</think>``. With vLLM's reasoning parser the
answer never contains thinking; if a server lacks the parser, Qwen3.5-style
templates (which prefill ``<think>``) leak the reasoning into the content, and the
cut both recovers the answer and lets the caller flag the misconfiguration.
The rules were checked against Li & Shirado's released replies (study README).
"""

from __future__ import annotations

import re

_THINK_CLOSE = "</think>"
_WRAP = r"[\s*_`\"'(\[#]*"
_UNWRAP = r"[\s*_`\"').\]!]*"
_LABEL = r"(?:(?:final\s+)?(?:answer|decision|choice|conclusion)\**\s*[:\-]\**\s*)?"
_OPTION_ANY = re.compile(r"\boption\s*\**\s*([AB])\b", re.IGNORECASE)
_OPTION_LINE = re.compile(
    rf"^{_WRAP}{_LABEL}{_WRAP}(?:option\s*)?\**\s*([AB]){_UNWRAP}$", re.IGNORECASE
)
# Answer-first replies: "# Option A", "Option B: Do nothing.", "**Option B** (keep the points)".
_OPTION_HEAD = re.compile(
    rf"^[\s*_`\"'#]*{_LABEL}[\s*_`\"'\[]*option\s*\**\s*([AB])\b\**\s*(?:$|[:(\-–—.,])",
    re.IGNORECASE,
)
# Evaluative words that mark a recommendation ("the safer choice is Option B").
_VERDICT = (
    r"(?:more\s+|most\s+)?(?:rational|best|optimal|safer|safest|logical|right|better|wise|wisest|"
    r"sensible|dominant|strategic|recommended|prudent|risk-averse|likely|preferable|preferred|"
    r"appropriate|reasonable|advisable|justified)"
)
_FIRST_PERSON = (
    r"(?:\bi(?:'ll|\s+will|\s+would)?\s+(?:choose|select|pick|go\s+with|opt\s+for|decide\s+on)"
    r"|\bi(?:'m|\s+am)\s+(?:choosing|selecting|picking|going\s+with|opting\s+for)"
    r"|\bi(?:'ve|\s+have)\s+(?:decided|chosen)"
    r"|\b(?:the|my)\s+(?:final\s+)?(?:choice|decision|answer|vote)\s+(?:is|would\s+be)"
    rf"|\b{_VERDICT}(?:\s+and\s+[\w-]+)?\**\s+(?:choice|decision|option|strategy|move|approach)"
    r"\**\s+(?:is|would\s+be|here\s+is)"
    r"|\bfinal\s+(?:answer|decision|choice)\b"
    r"|^[\s*#_]*(?:answer|decision|choice|conclusion)\**\s*:)"
)
_OPTION_STATEMENT = re.compile(
    rf"{_FIRST_PERSON}[^.]{{0,60}}?\boption\s*\**\s*([AB])\b"
    rf"|\boption\s*\**\s*([AB])\b[^.\n]{{0,70}}?\b(?:is|seems|appears)\s+(?:to\s+be\s+)?"
    rf"(?:the\s+|a\s+)?{_VERDICT}\b",
    re.IGNORECASE | re.MULTILINE,
)
_OPTION_EMPHASIS = re.compile(r"(?:\*\*|__)\s*(?:option\s*)?([AB])\s*(?:\*\*|__)", re.IGNORECASE)
# Hypotheticals and concessions ("if I choose Option A", "While Option B is the safer choice").
_CONDITIONAL = re.compile(
    r"\b(?:if|whether|suppose|unless|when|while|although|though)\s+(?:\w+\s+){0,2}\W{0,3}$",
    re.IGNORECASE,
)

_RANGE_ECHO = re.compile(r"\[\s*0\s*(?:to|-|–)\s*100\s*\]")
_NUMBER = r"(\d{1,3})(?![\d.,]\d|\s*%)"
_AMOUNT_LINE = re.compile(
    rf"^{_WRAP}(?:(?:final\s+)?(?:answer|decision|amount|offer|minimum(?:\s+acceptable\s+offer)?)"
    rf"\**\s*[:\-]\**\s*)?{_WRAP}{_NUMBER}(?:\s*points?)?{_UNWRAP}$",
    re.IGNORECASE,
)
_AMOUNT_STATEMENT = re.compile(
    r"(?:\bi(?:'ll|\s+will|\s+would)?\s+(?:decide\s+to\s+)?(?:give|allocate|offer|transfer|send|"
    r"accept|choose\s+to\s+give|set)"
    r"|\bi(?:'m|\s+am)\s+(?:giving|allocating|offering|transferring|sending|setting)"
    r"|\bi(?:'ve|\s+have)\s+decided\s+to\s+(?:give|allocate|offer|transfer|send|set)"
    r"|\bmy\s+(?:final\s+)?(?:minimum\s+acceptable\s+offer|minimum|answer|decision|choice|offer)"
    r"\s+(?:is|would\s+be)"
    r"|\bminimum\s+acceptable\s+offer\b[^\d\n.]{0,30}?(?::|\bis\b|\bof\b|\bat\b|\bwould\s+be\b)"
    r"|\bfinal\s+answer\b|^[\s*#_]*(?:answer|decision)\**\s*:)"
    rf"[^\d\n]{{0,25}}?{_NUMBER}",
    re.IGNORECASE | re.MULTILINE,
)
_AMOUNT_EMPHASIS = re.compile(rf"(?:\*\*|__)\s*{_NUMBER}|\\boxed\{{\s*{_NUMBER}\s*\}}")
_INTEGER = re.compile(rf"(?<![\d.]){_NUMBER}")


def strip_reasoning(text: str) -> tuple[str, bool]:
    """Return (answer text, whether reasoning markup was found in the reply)."""
    if _THINK_CLOSE in text:
        return text.rsplit(_THINK_CLOSE, 1)[1].strip(), True
    if "<think>" in text:  # opened but never closed: no answer
        return "", True
    return text.strip(), False


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _consistent(values: list[str]) -> str | None:
    return values[0] if values and len(set(values)) == 1 else None


def _statements(pattern: re.Pattern[str], text: str) -> list[str]:
    found = []
    for match in pattern.finditer(text):
        if _CONDITIONAL.search(text[max(0, match.start() - 40) : match.start()]):
            continue
        found.append(next(group for group in match.groups() if group))
    return found


def parse_option(text: str) -> str | None:
    """'A' or 'B', or None when absent or ambiguous."""
    answer = text.strip()
    if not answer:
        return None
    whole = _OPTION_LINE.match(" ".join(answer.split()))
    if whole:
        return whole.group(1).upper()
    lines = [m.group(1).upper() for line in _lines(answer) if (m := _OPTION_LINE.match(line))]
    if lines:
        return _consistent(lines)
    head = _OPTION_HEAD.match(_lines(answer)[0])
    if head:
        return head.group(1).upper()
    statements = [value.upper() for value in _statements(_OPTION_STATEMENT, answer)]
    if statements:
        return _consistent(statements) or statements[-1]
    mentioned = {value.upper() for value in _OPTION_ANY.findall(answer)}
    if len(mentioned) == 1:
        return mentioned.pop()
    emphasis = [value.upper() for value in _OPTION_EMPHASIS.findall(answer)]
    return _consistent(emphasis)


def parse_amount(text: str, *, low: int = 0, high: int = 100) -> int | None:
    """An integer in [low, high], or None when absent, ambiguous or out of range."""
    answer = _RANGE_ECHO.sub(" ", text.strip())
    if not answer:
        return None

    def valid(value: str | None) -> int | None:
        if value is None:
            return None
        number = int(value)
        return number if low <= number <= high else None

    whole = _AMOUNT_LINE.match(" ".join(answer.split()))
    if whole:
        return valid(whole.group(1))
    lines = [m.group(1) for line in _lines(answer) if (m := _AMOUNT_LINE.match(line))]
    if lines:
        return valid(_consistent(lines))
    statements = _statements(_AMOUNT_STATEMENT, answer)
    if statements:
        return valid(_consistent(statements) or statements[-1])
    emphasis = [a or b for a, b in _AMOUNT_EMPHASIS.findall(answer)]
    if emphasis:
        return valid(_consistent(emphasis))
    last = set(_INTEGER.findall(_lines(answer)[-1])) if _lines(answer) else set()
    if len(last) == 1:
        return valid(last.pop())
    numbers = set(_INTEGER.findall(answer))
    return valid(numbers.pop()) if len(numbers) == 1 else None
