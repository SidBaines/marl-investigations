"""Synthetic code problems keep benchmark text out of committed test fixtures."""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from typing import Any

from marli.envs.base import Task

SUM_SOLUTION = "a, b = map(int, input().split())\nprint(a + b)\n"


def private_payload(value: Any) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(value))).decode()


def lcb_row(
    *, functional: bool = False, contest_date: str = "2025-03-01T12:00:00"
) -> dict[str, Any]:
    kind = "functional" if functional else "stdin"
    public = [{"input": "2\n3" if functional else "2 3\n", "output": "5", "testtype": kind}]
    hidden = [{"input": "4\n5" if functional else "4 5\n", "output": "9", "testtype": kind}]
    return {
        "question_id": "synthetic-sum",
        "question_content": "Synthetic: add two integers.",
        "contest_date": contest_date,
        "platform": "leetcode" if functional else "atcoder",
        "difficulty": "easy",
        "starter_code": "class Solution:\n    def add(self, a, b):\n        pass\n"
        if functional
        else "",
        "public_test_cases": json.dumps(public),
        "private_test_cases": private_payload(json.dumps(hidden)),
        "metadata": json.dumps({"func_name": "add"} if functional else {}),
    }


def deepcoder_row(subset: str, *, functional: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"problem": "Synthetic: add two integers."}
    if subset == "taco":
        tests = (
            {"fn_name": "add", "inputs": [[2, 3], [4, 5]], "outputs": [[5], [9]]}
            if functional
            else {"inputs": ["2 3\n", "4 5\n"], "outputs": ["5\n", "9\n"]}
        )
    else:
        tests = [{"input": "2\n3" if functional else "2 3\n", "output": "5"}]
        if subset == "primeintellect":
            tests[0]["type"] = "stdin_stdout"
        elif subset == "lcbv5":
            tests[0]["testtype"] = "functional" if functional else "stdin"
            row["metadata"] = {"func_name": "add" if functional else None}
            row["starter_code"] = "def add(a, b):\n    pass\n" if functional else ""
    row["tests"] = json.dumps(tests)
    if subset in ("taco", "primeintellect"):
        row["solutions"] = ["REFERENCE_SOLUTION_MUST_NOT_LEAK"]
    return row


def code_task(*, functional: bool = False) -> Task:
    return Task(
        "synthetic/add",
        "Synthetic: add two integers.",
        {
            "kind": "functional" if functional else "stdin",
            "fn_name": "add" if functional else None,
            "tests": [
                {"input": "2\n3" if functional else "2 3\n", "output": "5"},
                {"input": "4\n5" if functional else "4 5\n", "output": "9"},
                {"input": "7\n8" if functional else "7 8\n", "output": "15"},
            ],
            "public": [{"input": "2\n3" if functional else "2 3\n", "output": "5"}],
        },
        {"source": "synthetic", "starter_code": "# Start here\n", "n_tests": 3},
    )
