"""The sacrifice-relay study's configs: recorded runs keep their identity; eval conditions compose.

Hashes were recorded on the commit before code_rules gained its transfer-eval settings
(m6-harness 2476da3). Run dirs add their input digests to these, so the recorded experiment 1 and
2 runs keep resuming as no-ops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_envs_code_rules_transfer import DEFAULTS, HELD_OUT, TRAINED, repo_task

from marli import config, verbs
from marli.envs.code_rules import TRAINING_RULE_FAMILIES, CodeRulesEnv
from marli.interact.configs import build_protocol, resolve_protocol

ROOT = Path(__file__).resolve().parents[1]
STUDY = "experiments/2026-09-25_sacrifice-relay"
EXP2 = f"{STUDY}/exp2_mandatory_rule/configs"
EVAL = f"{STUDY}/exp2_eval/configs"


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


CONDITIONS = sorted(path.stem for path in (ROOT / EVAL / "conditions").glob("*.yaml"))


def test_the_study_defines_every_planned_condition() -> None:
    planned = ["train_repos", "heldout", "new_rules", "reworded", "tools", "notes", "replies"]
    assert sorted([*planned, "n3", "n5", "far"]) == CONDITIONS


@pytest.mark.parametrize("name", CONDITIONS)
def test_transfer_conditions_compose_onto_the_training_setup(name: str) -> None:
    cfg = compose("eval rollout", [
        f"{STUDY}/configs/eval_common.yaml", f"{STUDY}/configs/eval_gate_n4.yaml",
        f"{EXP2}/env.yaml", f"{EXP2}/variants/checks.yaml", f"{EVAL}/eval.yaml",
        f"{EVAL}/conditions/{name}.yaml",
    ])
    assert cfg.episodes_per_task == 2 and cfg.run_seed == 0
    assert cfg.policies["q"].sampling.temperature == 1 and cfg.policies["q"].sampling.top_p == 1
    assert cfg.seating == {"contributor": "q"}
    _, relay, _ = resolve_protocol(cfg.protocol, cfg.protocol_config)
    n = relay.n_agents
    assert cfg.tasks is not None
    if name == "train_repos":
        assert cfg.tasks == f"{STUDY}/out/repos_n4/taskset.json"
    else:
        repos = "repos_n4_new" if name in ("new_rules", "far") else f"repos_n{n}"
        assert cfg.tasks == f"{STUDY}/exp2_eval/out/{repos}/taskset.json"
    families = HELD_OUT if "repos_n4_new" in cfg.tasks else TRAINING_RULE_FAMILIES
    env = CodeRulesEnv(cfg.env_config, repo_task(n, families))
    # Training's scoring, layout and prompt facts are kept in every condition.
    for key, value in TRAINED.items():
        assert getattr(env.config, key) == value, key
    assert env.config.others_note is False and env.config.notes == "visible"
    advertised = {tool.spec.name for tool in env.tools("contributor")}
    assert advertised <= set(relay.env_tools)
    build_protocol(cfg.protocol, cfg.protocol_config).adjust_limits(cfg.limits)
    assert cfg.limits.episode.max_gen_tokens == n * cfg.limits.agent.max_gen_tokens == n * 12288
    assert cfg.limits.agent.max_calls == 30 and cfg.limits.call.max_tokens == 6144
    assert cfg.limits.ctx.max_ctx == 24576
    changed = {key for key in DEFAULTS if getattr(env.config, key) != DEFAULTS[key]}
    expected = {
        "reworded": {"instructions"},
        "tools": {"tool_names", "tool_text"},
        "notes": {"notes_path"},
        "replies": {"ci_replies"},
        "far": set(DEFAULTS),
    }
    assert changed == expected.get(name, set())
    trained_prompt = (
        "You are a software engineer contributing to a shared code repository. "
        "Follow the instructions in the first message."
    )
    assert (relay.system_prompt == trained_prompt) == (name not in ("reworded", "far"))


def test_a3b_eval_server_adapts_exactly_the_trained_modules() -> None:
    from marli.model import load_model

    serve = compose("serve vllm", [f"{EVAL}/serve_eval_a3b.yaml"])
    trained = compose("serve vllm", [f"{EXP2}/serve_a3b.yaml"])
    model = load_model("qwen3_6_35b_a3b")
    assert serve.model == model.name
    assert serve.lora_target_modules == trained.lora_target_modules == model.lora["target_modules"]
    assert serve.max_lora_rank == 32 and serve.max_model_len == 32768
