"""Qwen3.5-MoE trains its registry targets only; routed experts and router stay frozen."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from _marli_tiny_models import tiny_qwen3_5_moe  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from marli.model import load_model  # noqa: E402
from marli.render.base import Msg  # noqa: E402
from marli.render.fake import FakeRenderer  # noqa: E402
from marli.train.backends.local import LocalLearner, LocalLearnerPool  # noqa: E402
from marli.train.types import LearnerSpec, TrainDatum  # noqa: E402

A3B = "qwen3_6_35b_a3b"
TRAINED = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
}


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_pool(model: Any = None, **spec_changes: Any) -> LocalLearnerPool:
    spec = load_model(A3B)
    if spec_changes:
        spec = replace(spec, **spec_changes)
    return LocalLearnerPool.from_model(
        tiny_qwen3_5_moe() if model is None else model,
        tokenizer_sha=FakeRenderer.tokenizer_sha,
        model_spec=spec,
    )


def make_learner(pool: LocalLearnerPool, name: str = "policy") -> LocalLearner:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(456)
        return LocalLearner(name, LearnerSpec(backend="local", rank=4, learning_rate=1e-3), pool)


def logprobs(learner: LocalLearner, tokens: tuple[int, ...]) -> torch.Tensor:
    model = learner.pool.peft_model
    model.set_adapter(learner.name)
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([tokens])
        logits = model(input_ids=ids[:, :-1], use_cache=False).logits.float()
        return logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze().clone()


def datum(learner: LocalLearner) -> TrainDatum:
    renderer = FakeRenderer()
    prompt = renderer.initial(None, (), [Msg(role="user", content="review the repo")])
    completion = renderer.encode_completion("ran ci_review and wrote NOTES.md")
    tokens = tuple(prompt + completion)
    mask = (0.0,) * (len(prompt) - 1) + (1.0,) * len(completion)
    sample = logprobs(learner, tokens).tolist()
    return TrainDatum(
        learner=learner.name,
        episode_id="episode",
        agent_id="contrib0",
        role="contributor",
        segment_id="segment",
        session_idx=0,
        policy_version=learner.version,
        tokens=tokens,
        logprobs=tuple(lp if m else 0.0 for lp, m in zip(sample, mask, strict=True)),
        mask=mask,
        advantages=mask,
    )


def frozen_moe(learner: LocalLearner) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in learner.pool.base_model.named_parameters()
        if ".mlp.experts." in name or ".mlp.gate." in name or "shared_expert_gate" in name
    }


def test_registry_targets_reach_attention_linear_attention_and_shared_expert_only() -> None:
    from peft.tuners.lora.layer import ParamWrapper

    learner = make_learner(make_pool())
    model = learner.pool.peft_model
    wrapped = {
        name.split(".layers.", 1)[1].split(".", 1)[1]
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and learner.name in module.lora_A
    }
    assert wrapped == TRAINED
    assert not any(isinstance(module, ParamWrapper) for module in model.modules())
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in name for name in trainable)
    # The routed experts (3-D), the router and the shared-expert gate are frozen base weights.
    assert frozen_moe(learner)
    assert not any(
        p.requires_grad
        for name, p in model.named_parameters()
        if ".experts." in name or ".mlp.gate." in name or "shared_expert_gate" in name
    )


async def test_step_zero_is_on_policy_and_updates_leave_experts_frozen() -> None:
    learner = make_learner(make_pool())
    d = datum(learner)
    before, experts = logprobs(learner, d.tokens), frozen_moe(learner)
    result = await learner.train_step([d])
    assert result.kl_sample_train == pytest.approx(0, abs=1e-6)
    assert result.metrics["ratio_mean"] == pytest.approx(1, abs=1e-6)
    assert result.grad_norm > 0
    assert not torch.equal(before, logprobs(learner, d.tokens))
    after = frozen_moe(learner)
    assert after.keys() == experts.keys()
    assert all(torch.equal(after[key], value) for key, value in experts.items())


async def test_export_uses_serving_keys_and_restores_exactly(tmp_path: Path) -> None:
    learner = make_learner(make_pool())
    d = datum(learner)
    await learner.train_step([d])
    state = Path(await learner.save_state(str(tmp_path / "state")))
    exported = state / "adapter"
    weights = load_file(exported / "adapter_model.safetensors")
    assert weights and all(
        key.startswith("base_model.model.model.language_model.layers.") for key in weights
    )
    assert not any(".experts." in key or ".mlp.gate." in key for key in weights)
    modules = {
        key.split(".layers.", 1)[1].split(".", 1)[1].rsplit(".lora_", 1)[0] for key in weights
    }
    assert modules == TRAINED
    config = json.loads((exported / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "Qwen/Qwen3.6-35B-A3B"
    assert not config.get("target_parameters")
    restored = make_learner(make_pool(), name="policy")
    await restored.load_state(str(state))
    torch.testing.assert_close(
        logprobs(restored, d.tokens), logprobs(learner, d.tokens), rtol=0, atol=0
    )


def test_fused_experts_require_explicit_targets_and_grouped_mm() -> None:
    with pytest.raises(ValueError, match="lora.target_modules explicitly"):
        make_pool(lora={})
    model = tiny_qwen3_5_moe()
    model.config._experts_implementation = "eager"
    with pytest.raises(ValueError, match="grouped_mm"):
        make_pool(model)
    # Expert LoRA through PEFT target_parameters stays a gpt-oss-only experiment.
    with pytest.raises(ValueError, match="only supported for gpt-oss"):
        LocalLearner(
            "policy",
            LearnerSpec(backend="local", rank=4, learning_rate=1e-3),
            make_pool(),
            target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            allow_unverified_expert_lora=True,
        )


def test_text_only_loader_requests_grouped_mm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def load(path: str, **kwargs: Any) -> Any:
        calls.append((path, kwargs))
        return tiny_qwen3_5_moe()

    monkeypatch.setattr(transformers.Qwen3_5MoeForCausalLM, "from_pretrained", load)
    pool = LocalLearnerPool(load_model(A3B), device="cpu", dtype="float32")
    assert calls == [
        (
            "Qwen/Qwen3.6-35B-A3B",
            {
                "dtype": torch.float32,
                "attn_implementation": "sdpa",
                "experts_implementation": "grouped_mm",
            },
        )
    ]
    assert pool.target_modules == tuple(load_model(A3B).lora["target_modules"])
