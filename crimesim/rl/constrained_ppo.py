"""Fairness-constrained PPO for the PolicingEnv.

Two methods, both shaping the reward with the per-shift enforcement disparity:

* **Penalty method** (``train_penalized_ppo``, the one used for the main results):
  a fixed weight ``w`` -- reward becomes ``r - w * disparity_shift``. Sweeping
  ``w`` traces the efficiency-equity frontier by construction; robust, no tuning.
  With ``fairness.penalty_mode="excess_log_ref"`` the penalty is instead
  ``w * max(0, ln disparity_shift - ln ref_disparity)``: only disparity beyond
  a reference policy's (the need-proportional GreedyOracle's) is penalised.

* **Lagrangian method** (``train_lagrangian_ppo``): dual ascent on a multiplier
  toward a disparity ceiling ``alpha`` (Chow et al. 2017; Tessler et al. 2019;
  Stooke et al. 2020). Reported as an alternative.

Both return ``(model, vecnormalize, history)``; ``history`` is empty for the
penalty method.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

from crimesim.config import Config
from crimesim.env.policing_env import PolicingEnv

_COVERAGE_REWARDS = ("captured", "recorded_coverage", "latent_coverage")


class ShapedRewardWrapper(VecEnvWrapper):
    """reward <- reward - weight * info[key].  ``weight`` may be updated in place."""

    def __init__(self, venv, weight: float, key: str):
        super().__init__(venv)
        self.weight = float(weight)
        self.key = key
        self._pen_run = np.zeros(venv.num_envs)
        self.step_pen: list[float] = []

    def reset(self):
        self._pen_run[:] = 0.0
        return self.venv.reset()

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        pen = np.array([float(i.get(self.key, 0.0)) for i in infos])
        self.step_pen.extend(pen.tolist())
        return obs, rew - self.weight * pen, dones, infos

    def drain(self) -> float:
        v = float(np.mean(self.step_pen)) if self.step_pen else 0.0
        self.step_pen.clear()
        return v


class DualAscentCallback(BaseCallback):
    """Projected dual ascent on the reward-shaping weight (the multiplier lambda).

    The env's ``cost`` is already hinged at the ceiling, ``max(0, disparity_cum -
    alpha)`` (``FairnessMeter``), so the constraint is E[cost] <= ``cost_limit``
    and lambda grows only while the rollout-mean cost exceeds it. (The previous
    version subtracted a hard-coded 0.15 instead, letting disparity settle above
    alpha -- PIVOT_AUDIT.md.)
    """

    def __init__(self, wrapper: ShapedRewardWrapper, lr: float, lambda_max: float,
                 cost_limit: float = 0.0, verbose: int = 0):
        super().__init__(verbose)
        self.w, self.lr, self.lambda_max, self.cost_limit = wrapper, lr, lambda_max, cost_limit
        self.history: list[tuple[int, float, float]] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        mean_cost = self.w.drain()                       # mean hinged disparity
        self.w.weight = float(np.clip(
            self.w.weight + self.lr * (mean_cost - self.cost_limit), 0.0, self.lambda_max))
        self.history.append((self.num_timesteps, mean_cost, self.w.weight))


def _venv(cfg: Config):
    def thunk(seed):
        def _f():
            e = PolicingEnv(cfg); e.reset(seed=seed); return Monitor(e)
        return _f
    return DummyVecEnv([thunk(cfg.rl.seed + i) for i in range(cfg.rl.n_envs)])


def _ppo(cfg: Config, env, verbose):
    rl = cfg.rl
    return PPO("MlpPolicy", env, n_steps=max(rl.n_steps // rl.n_envs, 32),
               batch_size=rl.batch_size, gamma=rl.gamma, gae_lambda=rl.gae_lambda,
               learning_rate=rl.learning_rate, ent_coef=rl.ent_coef,
               policy_kwargs=dict(net_arch=[128, 128]), seed=rl.seed, verbose=verbose)


def _finish(model, vnorm, save_dir, cfg, history, resumed_at=()):
    if save_dir is not None:
        from crimesim.rl.ppo import finish_resume, save_obs_rms
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir / "ppo_model")
        save_obs_rms(vnorm, save_dir / "obs_rms.npz")
        cfg.save(save_dir / "config.yaml")
        np.save(save_dir / "lambda_history.npy", np.array(history) if history else np.zeros((0, 3)))
        finish_resume(save_dir, list(resumed_at))
    return model, vnorm, history


_PENALTY_KEYS = {"linear": "disparity_shift", "excess_log_ref": "excess_log_disparity"}


def train_penalized_ppo(cfg: Config, *, save_dir=None, verbose: int = 0):
    f = cfg.fairness
    if f.penalty_mode not in _PENALTY_KEYS:
        raise ValueError(f"unknown penalty_mode={f.penalty_mode!r}")
    if f.penalty_mode == "excess_log_ref" and f.ref_disparity <= 0:
        raise ValueError("penalty_mode='excess_log_ref' needs fairness.ref_disparity > 0")
    from crimesim.rl.ppo import learn_resumable
    shaped = ShapedRewardWrapper(_venv(cfg), f.penalty, _PENALTY_KEYS[f.penalty_mode])
    norm_r = cfg.env.reward_on not in _COVERAGE_REWARDS
    model, vnorm, resumed_at = learn_resumable(
        cfg, shaped, lambda v: _ppo(cfg, v, verbose), norm_reward=norm_r,
        save_dir=save_dir, verbose=verbose)
    return _finish(model, vnorm, save_dir, cfg, [], resumed_at)


def train_lagrangian_ppo(cfg: Config, *, save_dir=None, verbose: int = 0):
    f = cfg.fairness
    from crimesim.rl.ppo import learn_resumable
    shaped = ShapedRewardWrapper(_venv(cfg), f.lambda_init, "cost")
    norm_r = cfg.env.reward_on not in _COVERAGE_REWARDS
    cb = DualAscentCallback(shaped, f.lambda_lr, f.lambda_max, f.cost_limit, verbose)

    def restore(state):                      # the multiplier and its history survive a resume
        shaped.weight = float(state["lambda"])
        cb.history[:] = [tuple(h) for h in state["lambda_history"]]

    model, vnorm, resumed_at = learn_resumable(
        cfg, shaped, lambda v: _ppo(cfg, v, verbose), norm_reward=norm_r, save_dir=save_dir,
        callbacks=[cb], extra_state=lambda: {"lambda": shaped.weight, "lambda_history": cb.history},
        restore_state=restore, verbose=verbose)
    return _finish(model, vnorm, save_dir, cfg, cb.history, resumed_at)
