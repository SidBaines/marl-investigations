"""Training must resolve built-ins even in a process that has never made an env."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cold_training_preflight_loads_builtin_envs_idempotently() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
from marli.envs.registry import ENVS, load_builtin_envs, make_env
from marli.train.loop import _preflight
from marli.train.rl import TrainRLConfig
from marli.train.types import LearnerSpec
assert 'marli.envs.code_rules' not in sys.modules
cfg = TrainRLConfig(
    env='code_rules', protocol='relay_n4',
    learners={'policy': LearnerSpec(base_model='qwen3_8b', backend='fake')},
    seating={'contributor': 'learner:policy'},
)
_preflight(cfg)
assert {'math', 'code_fn', 'code_rules'} <= set(ENVS.names())
factories = {name: ENVS.get(name) for name in ENVS.names()}
load_builtin_envs()
_preflight(cfg)
assert factories == {name: ENVS.get(name) for name in ENVS.names()}
from marli.envs.base import Task
env = make_env('math', {}, Task('synthetic', '2+3?', '5', {'answer_format': 'integer'}))
assert env.name == 'math'
assert not {'torch', 'tinker', 'transformers', 'datasets', 'vllm'} & sys.modules.keys()
"""],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
