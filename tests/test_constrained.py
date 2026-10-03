import numpy as np
import pytest

from crimesim.config import load_config
from crimesim.rl.constrained_ppo import DualAscentCallback


class _FakeWrapper:
    def __init__(self, costs):
        self.weight, self._costs = 0.0, list(costs)

    def drain(self):
        return self._costs.pop(0)


def _run(costs, lr=0.5, lambda_max=10.0, cost_limit=0.0):
    w = _FakeWrapper(costs)
    cb = DualAscentCallback(w, lr=lr, lambda_max=lambda_max, cost_limit=cost_limit)
    cb.num_timesteps = 0  # BaseCallback reads this from the model; none attached here
    lambdas = []
    for _ in costs:
        cb._on_rollout_end()
        lambdas.append(w.weight)
    return np.array(lambdas)


def test_lambda_rises_while_any_disparity_exceeds_ceiling():
    # cost = max(0, disparity_cum - alpha); a small positive cost must still push lambda up
    lam = _run([0.1, 0.1, 0.1])
    assert np.all(np.diff(np.r_[0.0, lam]) > 0)


def test_lambda_stays_at_zero_when_constraint_is_satisfied():
    assert np.all(_run([0.0, 0.0, 0.0]) == 0.0)


def test_lambda_falls_below_cost_limit_and_is_projected():
    lam = _run([2.0] + [0.0] * 10, lr=0.5, lambda_max=0.8, cost_limit=0.2)
    assert lam[0] == pytest.approx(0.8)                    # clipped at lambda_max
    assert lam[1] == pytest.approx(0.7)                    # 0.8 + 0.5 * (0 - 0.2)
    assert lam[-1] == 0.0                                  # never negative


def test_default_config_has_zero_cost_limit():
    assert load_config("configs/ppo_lagrangian.yaml").fairness.cost_limit == 0.0
