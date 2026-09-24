"""Tiny real PEFT models exercise isolation, bounded logits, and exact resume."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from peft import PeftModel, get_peft_model_state_dict  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from marli.render.base import Msg  # noqa: E402
from marli.render.fake import FakeRenderer  # noqa: E402
from marli.train.backends.local import LocalLearner, LocalLearnerPool  # noqa: E402
from marli.train.backends.local.learner import chunked_logprobs  # noqa: E402
from marli.train.types import LearnerSpec, TrainDatum  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model() -> Qwen3ForCausalLM:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        config = Qwen3Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=512,
            max_position_embeddings=128,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "sdpa"
        return Qwen3ForCausalLM(config)


def make_learner(
    name: str = "alice",
    *,
    pool: LocalLearnerPool | None = None,
    loss: str = "importance_sampling",
    **kwargs: Any,
) -> LocalLearner:
    if pool is None:
        pool = LocalLearnerPool.from_model(tiny_model(), tokenizer_sha=FakeRenderer.tokenizer_sha)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(456)
        return LocalLearner(
            name,
            LearnerSpec(backend="local", rank=4, learning_rate=1e-3, loss=loss),
            pool,
            **kwargs,
        )


def full_logprobs(learner: LocalLearner, tokens: tuple[int, ...]) -> torch.Tensor:
    model = learner.pool.peft_model
    model.set_adapter(learner.name)
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([tokens])
        logits = model(input_ids=ids[:, :-1], use_cache=False).logits.float()
        return logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze().clone()


def datum(learner: LocalLearner, text: str = "hi", answer: str = "ok") -> TrainDatum:
    renderer = FakeRenderer()
    prompt = renderer.initial(None, (), [Msg(role="user", content=text)])
    completion = renderer.encode_completion(answer)
    tokens = tuple(prompt + completion)
    mask = (0.0,) * (len(prompt) - 1) + (1.0,) * len(completion)
    sample = full_logprobs(learner, tokens).tolist()
    return TrainDatum(
        learner=learner.name,
        episode_id="episode",
        agent_id=learner.name,
        role="peer",
        segment_id="segment",
        session_idx=0,
        policy_version=learner.version,
        tokens=tokens,
        logprobs=tuple(lp if m else 0.0 for lp, m in zip(sample, mask, strict=True)),
        mask=mask,
        advantages=mask,
    )


def adapter_weights(learner: LocalLearner) -> dict[str, torch.Tensor]:
    return {
        key: tensor.detach().clone()
        for key, tensor in get_peft_model_state_dict(
            learner.pool.peft_model, adapter_name=learner.name, save_embedding_layers=False
        ).items()
    }


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 1024])
def test_chunked_logprobs_and_gradients_match_full_projection(chunk_size: int) -> None:
    learner = make_learner()
    d = datum(learner, answer="yes")
    ids = torch.tensor([d.tokens])
    decoder = learner.pool.base_model.get_decoder()
    with torch.no_grad():
        hidden = decoder(input_ids=ids[:, :-1], use_cache=False).last_hidden_state
    head = learner.pool.base_model.get_output_embeddings()
    h1 = hidden.detach().clone().requires_grad_()
    h2 = hidden.detach().clone().requires_grad_()
    lp = chunked_logprobs(h1, head, ids[:, 1:], chunk_size=chunk_size)
    full = head(h2).float().log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    torch.testing.assert_close(lp, full, atol=1e-5, rtol=1e-5)
    parameters = (h1, *learner.parameters[-2:])
    actual = torch.autograd.grad(lp.sum(), parameters)
    expected = torch.autograd.grad(full.sum(), (h2, *learner.parameters[-2:]))
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)


def test_chunking_does_not_retain_vocabulary_sized_activations() -> None:
    learner = make_learner()
    d = datum(learner)
    ids = torch.tensor([d.tokens])
    with torch.no_grad():
        hidden = learner.pool.base_model.get_decoder()(
            input_ids=ids[:, :-1], use_cache=False
        ).last_hidden_state
    hidden.requires_grad_()
    saved_shapes = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        lp = chunked_logprobs(
            hidden, learner.pool.base_model.get_output_embeddings(), ids[:, 1:], chunk_size=3
        )
    assert not any(shape and shape[-1] == 512 for shape in saved_shapes)
    lp.sum().backward()
    assert hidden.grad is not None


@pytest.mark.parametrize("loss", ["importance_sampling", "ppo"])
async def test_step_zero_ratios_are_one_for_each_adapter(loss: str) -> None:
    first = make_learner(loss=loss)
    second = make_learner("bob", pool=first.pool, loss=loss)
    for learner in (first, second):
        d = datum(learner)
        result = await learner.train_step([d])
        assert result.kl_sample_train == pytest.approx(0, abs=1e-6)
        assert result.metrics["ratio_mean"] == pytest.approx(1, abs=1e-6)
        assert result.metrics["ratio_max"] == pytest.approx(1, abs=1e-6)
        assert result.loss == pytest.approx(-d.n_action_tokens, abs=1e-5)
        assert result.n_tokens == len(d.tokens) - 1
        assert result.n_action_tokens == d.n_action_tokens
        assert result.grad_norm > 0
        assert learner.step == 1 and learner.version == 0


async def test_adapters_outputs_weights_and_optimizers_are_isolated() -> None:
    first = make_learner()
    second = make_learner("bob", pool=first.pool)
    da, db = datum(first), datum(second)
    base_before = {
        name: p.detach().clone()
        for name, p in first.pool.base_model.named_parameters()
        if "lora_" not in name
    }
    bob_before, output_before = adapter_weights(second), full_logprobs(second, db.tokens)
    assert set(map(id, first.parameters)).isdisjoint(map(id, second.parameters))
    await first.train_step([da])
    assert first.optimizer.state and not second.optimizer.state
    for key, value in adapter_weights(second).items():
        torch.testing.assert_close(value, bob_before[key], rtol=0, atol=0)
    torch.testing.assert_close(full_logprobs(second, db.tokens), output_before, rtol=0, atol=0)
    alice_before, alice_state = adapter_weights(first), copy.deepcopy(first.optimizer.state_dict())
    await second.train_step([db])
    for key, value in adapter_weights(first).items():
        torch.testing.assert_close(value, alice_before[key], rtol=0, atol=0)
    for key, state in first.optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, alice_state["state"][key][field], rtol=0, atol=0)
    for name, value in first.pool.base_model.named_parameters():
        if name in base_before:
            torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("loss", ["importance_sampling", "ppo"])
async def test_microbatch_invariance_compares_actual_gradients(loss: str) -> None:
    batched = make_learner(max_tokens_per_microbatch=16384, loss=loss)
    split = make_learner(max_tokens_per_microbatch=1, loss=loss)
    data = [datum(batched, answer="one"), datum(batched, text="longer question", answer="two")]
    captured: list[list[torch.Tensor]] = []

    def capture(optimizer: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
        captured.append(
            [p.grad.clone() for group in optimizer.param_groups for p in group["params"]]
        )

    for learner in (batched, split):
        learner.optimizer.register_step_pre_hook(capture)
    one = await batched.train_step(data)
    two = await split.train_step(data)
    assert one.loss == pytest.approx(two.loss, abs=1e-5)
    assert one.grad_norm == pytest.approx(two.grad_norm, rel=1e-5)
    for left, right in zip(*captured, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-4)
    for key, value in adapter_weights(batched).items():
        torch.testing.assert_close(value, adapter_weights(split)[key], atol=2e-6, rtol=1e-4)


async def test_save_load_resume_and_plain_peft_export(tmp_path: Path) -> None:
    learner = make_learner(alpha=8)
    d = datum(learner)
    await learner.train_step([d])
    before = full_logprobs(learner, d.tokens)
    path = Path(await learner.save_state(str(tmp_path / "states")))
    another_path = Path(await learner.save_state(str(tmp_path / "states")))
    assert path != another_path and path.name.startswith("step-00000001-")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["base_model"] == learner.model.hf_id
    assert len(manifest["git_commit"]) == 40
    assert manifest["rank"] == 4 and manifest["alpha"] == 8
    assert "lm_head" in manifest["target_modules"]
    restored = make_learner("restored", alpha=8)
    await restored.load_state(str(path))
    assert restored.step == 1
    torch.testing.assert_close(full_logprobs(restored, d.tokens), before, atol=0, rtol=0)
    await learner.train_step([d])
    await restored.train_step([replace(d, learner="restored")])
    torch.testing.assert_close(
        full_logprobs(restored, d.tokens), full_logprobs(learner, d.tokens), atol=0, rtol=0
    )
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    exported = Path(await restored.save_adapter(export_dir))
    assert (exported / "adapter_model.safetensors").is_file()
    config = json.loads((exported / "adapter_config.json").read_text())
    assert config["r"] == 4 and config["lora_alpha"] == 8
    assert set(config["target_modules"]) == set(manifest["target_modules"])
    plain = PeftModel.from_pretrained(tiny_model(), exported, local_files_only=True).eval()
    ids = torch.tensor([d.tokens])
    with torch.no_grad():
        logprobs = plain(ids[:, :-1]).logits.float().log_softmax(-1)
        logprobs = logprobs.gather(-1, ids[:, 1:, None]).squeeze()
    torch.testing.assert_close(logprobs, full_logprobs(restored, d.tokens), atol=0, rtol=0)
    await restored.load_state(str(path), with_optimizer=False)
    assert not restored.optimizer.state and restored.step == 0
    torch.testing.assert_close(full_logprobs(restored, d.tokens), before, atol=0, rtol=0)


async def test_positive_advantage_loss_decreases() -> None:
    learner = make_learner()
    d = datum(learner)
    results = [await learner.train_step([d]) for _ in range(4)]
    assert all(right.loss < left.loss for left, right in zip(results, results[1:], strict=False))


def test_optimizer_hyperparameters_come_from_spec() -> None:
    pool = LocalLearnerPool.from_model(tiny_model(), tokenizer_sha=FakeRenderer.tokenizer_sha)
    spec = LearnerSpec(
        backend="local",
        rank=2,
        learning_rate=0.004,
        beta1=0.7,
        beta2=0.8,
        eps=1e-6,
        weight_decay=0.02,
    )
    learner = LocalLearner("custom", spec, pool)
    group = learner.optimizer.param_groups[0]
    assert group["lr"] == spec.learning_rate
    assert group["betas"] == (0.7, 0.8)
    assert group["eps"] == 1e-6 and group["weight_decay"] == 0.02


async def test_checkpoint_mismatch_is_rejected_before_modifying_adapter(tmp_path: Path) -> None:
    learner = make_learner()
    d = datum(learner)
    path = await learner.save_state(str(tmp_path))
    await learner.train_step([d])
    before = adapter_weights(learner)
    manifest_path = Path(path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rank"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rank mismatch"):
        await learner.load_state(path)
    for key, value in adapter_weights(learner).items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


async def test_validation_and_sampler_stubs() -> None:
    learner = make_learner()
    d = datum(learner)
    for data, message in (
        ([], "non-empty"),
        ([replace(d, learner="wrong")], "routed"),
        ([replace(d, policy_version=1)], "current policy"),
        ([d, replace(d, policy_version=None)], "current policy"),
    ):
        with pytest.raises(ValueError, match=message):
            await learner.train_step(data)
    learner.model = replace(learner.model, max_ctx=2)
    with pytest.raises(ValueError, match="max ctx"):
        await learner.train_step([d])
    with pytest.raises(ValueError, match="backend"):
        LocalLearner("invalid", LearnerSpec(), learner.pool)
    with pytest.raises(ValueError, match="already exists"):
        make_learner(pool=learner.pool)
    with pytest.raises(NotImplementedError, match="M4-2"):
        learner.policy()
    with pytest.raises(NotImplementedError, match="M4-2"):
        await learner.sync_sampler("snapshot")
    await learner.close()


async def test_grad_clip_reports_preclip_norm_and_learning_rate_override() -> None:
    learner = make_learner()
    learner.spec.grad_clip = 0.01
    d = datum(learner)
    before = adapter_weights(learner)
    norms = []

    def capture(optimizer: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
        norms.append(torch.stack([p.grad.norm() for p in learner.parameters]).norm().item())

    learner.optimizer.register_step_pre_hook(capture)
    result = await learner.train_step([d], learning_rate=0.0)
    assert result.grad_norm > 0.01
    assert norms[0] == pytest.approx(0.01, rel=1e-4)
    for key, value in adapter_weights(learner).items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert all(p.grad is None for p in learner.parameters)


async def test_concurrent_steps_serialize_without_blocking_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = make_learner()
    second = make_learner("bob", pool=first.pool)
    da, db = datum(first), datum(second)
    entered, release = threading.Event(), threading.Event()
    original = first._train_step
    main_thread = threading.get_ident()

    def delayed(data: Any, learning_rate: float | None) -> Any:
        assert threading.get_ident() != main_thread
        entered.set()
        assert release.wait(5)
        return original(data, learning_rate)

    monkeypatch.setattr(first, "_train_step", delayed)
    one = asyncio.create_task(first.train_step([da]))
    two = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        one.cancel()
        two = asyncio.create_task(second.train_step([db]))
        await asyncio.sleep(0)
        assert first.pool.lock.locked() and second.step == 0 and not one.done()
        one.cancel()
        await asyncio.sleep(0)
        assert first.pool.lock.locked() and second.step == 0 and not one.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await one
        if two is not None:
            await two
    assert first.step == second.step == 1


def test_pool_loader_loads_base_once_and_requires_explicit_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = LocalLearnerPool.from_model(tiny_model(), tokenizer_sha="test").model
    calls = []

    def load(path: str, **kwargs: Any) -> Qwen3ForCausalLM:
        calls.append((path, kwargs))
        return tiny_model()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(Qwen3ForCausalLM, "from_pretrained", load)
    with pytest.raises(ValueError, match="CUDA"):
        LocalLearnerPool(spec, device="cuda")
    assert not calls
    pool = LocalLearnerPool(spec, device="cpu", dtype="float32")
    make_learner(pool=pool)
    make_learner("bob", pool=pool)
    assert calls == [(spec.hf_id, {"dtype": torch.float32, "attn_implementation": "sdpa"})]
    assert pool.base_model.is_gradient_checkpointing


def test_package_import_does_not_import_heavy_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import marli.train.backends.local; "
            "assert not ({'torch', 'peft', 'transformers'} & sys.modules.keys())",
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


async def test_mapped_adapter_round_trip_and_checkpoint_restore(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    from marli.train.backends.local.learner import _map_adapter_keys

    original = make_learner()
    mapping = {"base_model.model.model.layers.": "base_model.model.model.language_model.layers."}
    spec = replace(original.model, lora={"export_key_map": mapping})
    pool = LocalLearnerPool.from_model(
        tiny_model(), tokenizer_sha=FakeRenderer.tokenizer_sha, model_spec=spec
    )
    learner = make_learner(pool=pool)
    d = datum(learner)
    await learner.train_step([d])
    expected = adapter_weights(learner)
    state = Path(await learner.save_state(str(tmp_path / "state")))
    exported = state / "adapter"
    manifest = json.loads((exported / "manifest.json").read_text())
    assert manifest["export_key_map"] == mapping
    weights = load_file(exported / "adapter_model.safetensors")
    assert any(".language_model.layers." in key for key in weights)
    assert not any(key.startswith("base_model.model.model.layers.") for key in weights)
    reversed_weights = _map_adapter_keys(weights, {new: old for old, new in mapping.items()})
    assert reversed_weights.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(reversed_weights[key], value, rtol=0, atol=0)
    restored = make_learner(
        "restored",
        pool=LocalLearnerPool.from_model(
            tiny_model(),
            tokenizer_sha=FakeRenderer.tokenizer_sha,
            model_spec=spec,
        ),
    )
    await restored.load_state(str(state))
    torch.testing.assert_close(
        full_logprobs(restored, d.tokens), full_logprobs(learner, d.tokens), rtol=0, atol=0
    )
    assert restored.step == 1 and restored.optimizer.state
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    (plain_dir / "adapter_config.json").write_bytes((exported / "adapter_config.json").read_bytes())
    save_file(reversed_weights, plain_dir / "adapter_model.safetensors")
    plain = PeftModel.from_pretrained(tiny_model(), plain_dir, local_files_only=True).eval()
    ids = torch.tensor([d.tokens])
    with torch.no_grad():
        scores = plain(ids[:, :-1]).logits.float().log_softmax(-1)
        scores = scores.gather(-1, ids[:, 1:, None]).flatten()
    torch.testing.assert_close(scores, full_logprobs(learner, d.tokens), rtol=0, atol=0)


async def test_training_only_projects_checkpointed_chunks() -> None:
    learner = make_learner(logprob_chunk_size=3)
    d = datum(learner)
    shapes = []

    def record(module: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        shapes.append(tuple(output.shape))

    hook = learner.pool.base_model.get_output_embeddings().register_forward_hook(record)
    try:
        await learner.train_step([d])
    finally:
        hook.remove()
    chunks = (len(d.tokens) - 2) // 3 + 1
    # Each head projection runs once in forward and once in checkpoint recomputation.
    assert len(shapes) == 2 * chunks
    assert all(len(shape) == 2 and shape[0] <= 3 and shape[1] == 512 for shape in shapes)
    assert learner.logprob_chunk_size == 3 and make_learner().logprob_chunk_size == 1024


async def test_probe_scores_selected_adapter_and_frozen_base() -> None:
    first = make_learner()
    second = make_learner("second", pool=first.pool)
    d = datum(first)
    initial = full_logprobs(first, d.tokens)
    await first.train_step([d])
    adapted = full_logprobs(first, d.tokens)
    assert not torch.equal(initial, adapted)
    full_logprobs(second, d.tokens)  # Leave the other adapter selected before the probe.
    actual, base = await first.probe_logprobs(d.tokens)
    torch.testing.assert_close(torch.tensor(actual), adapted, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.tensor(base), initial, rtol=1e-6, atol=1e-6)


def test_registry_targets_override_all_linear_fallback() -> None:
    original = make_learner().model
    spec = replace(original, lora={"target_modules": ["q_proj", "v_proj"]})
    pool = LocalLearnerPool.from_model(tiny_model(), tokenizer_sha="test", model_spec=spec)
    learner = make_learner(pool=pool)
    assert pool.target_modules == ("q_proj", "v_proj")
    assert all(".q_proj." in key or ".v_proj." in key for key in adapter_weights(learner))


def test_qwen_text_loader_and_gpt_oss_flex_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    from marli.model import load_model

    calls = []

    def load(path: str, **kwargs: Any) -> Qwen3ForCausalLM:
        calls.append((path, kwargs))
        return tiny_model()

    monkeypatch.setattr(transformers.Qwen3_5ForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(transformers.GptOssForCausalLM, "from_pretrained", load)
    LocalLearnerPool(load_model("qwen3_5_4b"), device="cpu", dtype="float32")
    LocalLearnerPool(load_model("gpt_oss_20b"), device="cpu", dtype="float32")
    assert calls == [
        ("Qwen/Qwen3.5-4B", {"dtype": torch.float32, "attn_implementation": "sdpa"}),
        ("openai/gpt-oss-20b", {"dtype": torch.float32, "attn_implementation": "flex_attention"}),
    ]


def test_gpt_oss_expert_lora_requires_explicit_override() -> None:
    from marli.model import load_model

    model = tiny_model()
    model.model.mlp = torch.nn.Module()
    model.model.mlp.experts = torch.nn.Module()
    model.model.mlp.experts.register_parameter(
        "gate_up_proj", torch.nn.Parameter(torch.rand(2, 8, 8))
    )
    model.model.mlp.experts.register_parameter("down_proj", torch.nn.Parameter(torch.rand(2, 8, 8)))
    pool = LocalLearnerPool.from_model(
        model, tokenizer_sha="test", model_spec=load_model("gpt_oss_20b")
    )
    targets = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    with pytest.raises(ValueError, match="allow_unverified_expert_lora=True"):
        make_learner(pool=pool, target_parameters=targets)
    learner = make_learner(pool=pool, target_parameters=targets, allow_unverified_expert_lora=True)
    assert learner.pool.peft_model.peft_config[learner.name].target_parameters == targets
