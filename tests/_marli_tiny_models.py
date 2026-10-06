"""Deterministic tiny text models for local-learner tests, importable by spawned ranks."""

from __future__ import annotations

import multiprocessing
from typing import Any


def tiny_qwen3() -> Any:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

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


def tiny_qwen3_other_seed() -> Any:
    """Same architecture, different weights: ranks must refuse to train together."""
    import torch

    model = tiny_qwen3()
    if multiprocessing.parent_process() is not None:
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
    return model


def tiny_qwen3_fails_in_worker() -> Any:
    if multiprocessing.parent_process() is not None:
        raise RuntimeError("simulated worker load failure")
    return tiny_qwen3()


def tiny_qwen3_5_moe() -> Any:
    """Qwen3.5-MoE text model: one linear-attention and one full-attention layer, fused experts."""
    import torch
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        config = Qwen3_5MoeTextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            vocab_size=512,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "sdpa"
        return Qwen3_5MoeForCausalLM(config).float()
