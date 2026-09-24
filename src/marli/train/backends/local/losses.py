"""Sum-reduced objectives keep credit scaling identical to the Tinker backend.

The SDK 0.30.1 ``tinker/types/loss_fn_type.py`` names these objectives;
``tinker_cookbook/rl/train.py::train_step`` (0.5.7) passes logprobs and
advantages after ``_remove_mask``. Tinker's IS objective is
``-(exp(target_logprobs - sampler_logprobs) * advantages).sum()`` (also
recorded in ``docs/plans/2026-09-23-core-infra.md``, Loss semantics).
There is no RL mask argument on that wire: off-action advantages are zero.
Here the sidecar mask also excludes observations from diagnostics.

The installed SDK exposes PPO ``clip_low_threshold`` / ``clip_high_threshold``
but does not ship server-side defaults. The conventional thresholds 0.8/1.2
are expressed here as epsilon kwargs 0.2/0.2; server-default parity still
requires verification against the Tinker service documentation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _active(
    target_logprobs: Tensor, sampler_logprobs: Tensor, advantages: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if target_logprobs.ndim != 1 or not (
        target_logprobs.shape == sampler_logprobs.shape == advantages.shape == mask.shape
    ):
        raise ValueError("loss inputs must be aligned one-dimensional tensors")
    active = mask > 0
    # Select before exponentiating: masked observations must not introduce inf * 0.
    return (target_logprobs[active], sampler_logprobs[active], advantages[active], mask[active])


def _metrics(lp: Tensor, sample: Tensor, weight: Tensor) -> dict[str, Tensor]:
    with torch.no_grad():
        if not lp.numel():
            return {key: lp.new_zeros(()) for key in ("kl_sample_train", "ratio_mean", "ratio_max")}
        ratio = (lp - sample).exp()
        return {
            "kl_sample_train": ((sample - lp) * weight).sum() / weight.sum(),
            "ratio_mean": (ratio * weight).sum() / weight.sum(),
            "ratio_max": ratio.max(),
        }


def importance_sampling(
    target_logprobs: Tensor, sampler_logprobs: Tensor, advantages: Tensor, mask: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    """Tinker's negative importance-weighted advantage sum; see module sources."""
    lp, sample, advantage, weight = _active(target_logprobs, sampler_logprobs, advantages, mask)
    loss = -(weight * (lp - sample).exp() * advantage).sum()
    return loss, _metrics(lp, sample, weight)


def ppo(
    target_logprobs: Tensor,
    sampler_logprobs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    epsilon_lo: float = 0.2,
    epsilon_hi: float = 0.2,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sum the clipped surrogate; ``clip_frac`` counts ratios outside the bounds."""
    if not 0 <= epsilon_lo < 1 or epsilon_hi < 0:
        raise ValueError("PPO requires 0 <= epsilon_lo < 1 and epsilon_hi >= 0")
    lp, sample, advantage, weight = _active(target_logprobs, sampler_logprobs, advantages, mask)
    ratio = (lp - sample).exp()
    clipped = ratio.clamp(1 - epsilon_lo, 1 + epsilon_hi)
    loss = -(weight * torch.minimum(ratio * advantage, clipped * advantage)).sum()
    metrics = _metrics(lp, sample, weight)
    with torch.no_grad():
        outside = (ratio < 1 - epsilon_lo) | (ratio > 1 + epsilon_hi)
        metrics["clip_frac"] = (
            (outside * weight).sum() / weight.sum() if lp.numel() else lp.new_zeros(())
        )
    return loss, metrics


def cross_entropy(
    target_logprobs: Tensor, sampler_logprobs: Tensor, advantages: Tensor, mask: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    """SFT uses the mask as weights, with no dependence on advantages."""
    lp, sample, _, weight = _active(target_logprobs, sampler_logprobs, advantages, mask)
    return -(weight * lp).sum(), _metrics(lp, sample, weight)
