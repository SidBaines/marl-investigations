"""The sacrifice-relay study's configs: recorded runs keep their identity; eval conditions compose.

Hashes were recorded on the commit before code_rules gained its transfer-eval settings
(m6-harness 2476da3). Run dirs add their input digests to these, so the recorded experiment 1 and
2 runs keep resuming as no-ops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from marli import config, verbs

ROOT = Path(__file__).resolve().parents[1]
STUDY = "experiments/2026-09-25_sacrifice-relay"
EXP2 = f"{STUDY}/exp2_mandatory_rule/configs"


def compose(verb: str, layers: list[str]) -> Any:
    cls = verbs.resolve(verbs.get_verb(verb))[1]
    args = [layer if "=" in layer else str(ROOT / layer) for layer in layers]
    return config.parse(cls, args)


# (verb, config layers, the hash recorded before the transfer-eval settings existed)
HASHES = {
    "gate": (
        "eval rollout",
        [f"{STUDY}/configs/eval_common.yaml", f"{STUDY}/configs/eval_gate_n4.yaml",
         f"{EXP2}/env.yaml", f"{EXP2}/gate.yaml", f"{EXP2}/variants/checks.yaml"],
        "f3dfbc918571076d29a247f45e794104106415c6cb9a5d59220fe8a78e897289",
    ),
    "filter": (
        "eval rollout",
        [f"{STUDY}/configs/eval_common.yaml", f"{STUDY}/configs/eval_filter.yaml"],
        "a036dad5fd1e72f05297526afb322c5b29f4290ac968d8ba5a486d00f5437199",
    ),
    "train_team": (
        "train rl",
        [f"{STUDY}/configs/base.yaml", f"{EXP2}/env.yaml", f"{EXP2}/variants/checks.yaml",
         f"{EXP2}/train_team.yaml", "max_usd=1"],
        "f011b727ef2481b0c14cbb70cfe98e6e75da273aa92ffde9aab77e43c6115073",
    ),
    "train_individual": (
        "train rl",
        [f"{STUDY}/configs/base.yaml", f"{EXP2}/env.yaml", f"{EXP2}/variants/checks.yaml",
         f"{EXP2}/train_individual.yaml", "max_usd=1"],
        "a0a3f91be9d9ebc9bdf723d16c2dcc694a63a48c9a7af2d583f2ee276815fc84",
    ),
    "train_opener": (
        "train rl",
        [f"{STUDY}/configs/base.yaml", f"{EXP2}/env.yaml", f"{EXP2}/variants/checks.yaml",
         f"{EXP2}/train_opener.yaml", "max_usd=1"],
        "09052cf93ac214095decad128e84ff181702932edc217f20c4394a6ff797a84a",
    ),
    "train_team_a3b": (
        "train rl",
        [f"{STUDY}/configs/base.yaml", f"{EXP2}/env.yaml", f"{EXP2}/variants/checks.yaml",
         f"{EXP2}/train_team_a3b.yaml", "max_usd=1"],
        "ea9c36843333633811b1c428760a5b0f1e44d3a1ab609c26d8738b903f3b0028",
    ),
    "train_opener_a3b": (
        "train rl",
        [f"{STUDY}/configs/base.yaml", f"{EXP2}/env.yaml", f"{EXP2}/variants/checks.yaml",
         f"{EXP2}/train_opener_a3b.yaml", "max_usd=1"],
        "5e5bffa2453d5edd236da84b93b103580ab99fe59d94bd3e2aa9157f37049ee6",
    ),
    "repos_n4": (
        "data repos",
        ["tasks=pool/taskset.json", "n_per_repo=4", "seed=0"],
        "e1c2b4e6b4332a75f728401f07c7523ae6230e6d4a659f2ab70fd28bab6750f8",
    ),
    "candidates": (
        "data build",
        ["source=deepcoder", "max_n=800", "shuffle=true", "seed=0"],
        "94cf5e57c11419dc27f15b39273055c03442f389b1e15eca03603f6c358b0f77",
    ),
}


@pytest.mark.parametrize("name", sorted(HASHES))
def test_existing_sacrifice_relay_configs_keep_their_hashes(name: str) -> None:
    verb, layers, expected = HASHES[name]
    assert config.config_hash(compose(verb, layers)) == expected
