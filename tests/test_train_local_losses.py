"""Hand calculations and finite differences pin the backend's loss convention."""

from __future__ import annotations

from collections.abc import Callable

import pytest

torch = pytest.importorskip("torch")

from marli.train.backends.local import losses  # noqa: E402


def test_importance_sampling_sum_and_diagnostics() -> None:
    sample = torch.full((5,), -3.0, dtype=torch.float64)
    ratio = torch.tensor([0.5, 1, 1.5, 0.5, 20], dtype=torch.float64)
    lp = (sample + ratio.log()).requires_grad_()
    advantage = torch.tensor([2, 3, 4, -2, 100], dtype=torch.float64)
    mask = torch.tensor([1, 1, 1, 1, 0], dtype=torch.float64)
    loss, metrics = losses.importance_sampling(lp, sample, advantage, mask)
    assert loss.item() == pytest.approx(-9)
    assert metrics["ratio_mean"].item() == pytest.approx(0.875)
    assert metrics["ratio_max"].item() == pytest.approx(1.5)
    assert metrics["kl_sample_train"].item() == pytest.approx(-ratio[:4].log().mean().item())
    loss.backward()
    torch.testing.assert_close(lp.grad, torch.tensor([-1, -3, -6, 1, 0], dtype=torch.float64))
    assert all(not value.requires_grad for value in metrics.values())


def test_ppo_positive_and_negative_advantages_clip_in_opposite_directions() -> None:
    sample = torch.full((5,), -3.0, dtype=torch.float64)
    ratio = torch.tensor([0.5, 1, 1.5, 0.5, 20], dtype=torch.float64)
    lp = (sample + ratio.log()).requires_grad_()
    advantage = torch.tensor([2, 3, 4, -2, 100], dtype=torch.float64)
    mask = torch.tensor([1, 1, 1, 1, 0], dtype=torch.float64)
    loss, metrics = losses.ppo(lp, sample, advantage, mask)
    assert loss.item() == pytest.approx(-7.2)
    assert metrics["clip_frac"].item() == pytest.approx(0.75)
    loss.backward()
    torch.testing.assert_close(lp.grad, torch.tensor([-1, -3, 0, 0, 0], dtype=torch.float64))
    asymmetric, _ = losses.ppo(lp, sample, advantage, mask, epsilon_lo=0.1, epsilon_hi=0.3)
    assert asymmetric.item() == pytest.approx(-7.4)


def test_cross_entropy_uses_mask_as_weights() -> None:
    lp = torch.tensor([-0.5, -1, -2, -4], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([0.5, 1, 0, 1], dtype=torch.float64)
    loss, metrics = losses.cross_entropy(lp, lp.detach(), torch.zeros_like(lp), mask)
    assert loss.item() == pytest.approx(5.25)
    assert metrics["ratio_mean"].item() == 1
    assert metrics["kl_sample_train"].item() == 0
    loss.backward()
    torch.testing.assert_close(lp.grad, -mask)


@pytest.mark.parametrize(
    "objective", [losses.importance_sampling, losses.ppo, losses.cross_entropy]
)
def test_gradcheck(objective: Callable) -> None:
    sample = torch.full((4,), -2.0, dtype=torch.float64)
    lp = (sample + torch.tensor([0.5, 0.9, 1.5, 0.6]).double().log()).requires_grad_()
    advantage = torch.tensor([-0.8, 1.3, 0.4, -2.0], dtype=torch.float64)
    mask = torch.tensor([1, 0.5, 1, 0], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda p: objective(p, sample, advantage, mask)[0], (lp,))


@pytest.mark.parametrize(
    "objective", [losses.importance_sampling, losses.ppo, losses.cross_entropy]
)
def test_empty_action_mask_has_zero_loss_gradient_and_finite_metrics(objective: Callable) -> None:
    lp = torch.tensor([1000.0, -3.0], requires_grad=True)
    loss, metrics = objective(lp, torch.zeros(2), torch.ones(2), torch.zeros(2))
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(lp.grad, torch.zeros_like(lp))
    assert all(value.item() == 0 for value in metrics.values())


def test_weighted_metrics_and_masked_overflow() -> None:
    lp = torch.tensor([0.0, 0.69314718056, 1000.0], dtype=torch.float64, requires_grad=True)
    loss, metrics = losses.importance_sampling(
        lp, torch.zeros_like(lp), torch.ones_like(lp), torch.tensor([0.25, 0.75, 0])
    )
    assert loss.item() == pytest.approx(-1.75)
    assert metrics["ratio_mean"].item() == pytest.approx(1.75)
    loss.backward()
    assert torch.isfinite(lp.grad).all()
