"""``PolicingEnv`` -- a Gymnasium environment wrapping the emergent-crime simulator.

Each step is one patrol shift.

**Action.** By default (``action_mode="rule"``) the policy outputs the parameters
of an allocation rule -- how hard to concentrate on recently recorded crime, how
much to tilt toward high-hardship districts, and how much to mix back toward a
uniform allocation. This is low-dimensional, learnable in a noisy environment,
and a faithful model of a tuned predictive-policing system. ``action_mode="direct"``
instead lets the policy output a raw logit per district (kept for an ablation).

**Transition.** The ABM advances one shift under the resulting deployment.
Deterrence lowers latent crime in a patrolled district; detection raises the
share of its latent crime that is recorded -- the feedback channel.

**Observation.** An exponentially-weighted moving average of *recorded* crime per
district, the last deployment, time-of-day, and static district covariates. The
agent never sees latent crime unless ``observe_latent`` is set (oracle only).

**Reward.** ``recorded_coverage`` (the deployable objective: cover where crime is
found) by default; ``latent_coverage`` for the oracle; ``info["cost"]`` carries
the per-shift enforcement-disparity violation for the constrained learner.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crimesim.abm.generator import CrimeGenerator
from crimesim.abm.regions import region_for
from crimesim.config import Config
from crimesim.env.detection import record
from crimesim.env.fairness import FairnessMeter

EWMA_HALFLIFE = 5.0


def largest_remainder(shares: np.ndarray, total: int) -> np.ndarray:
    """Integer allocation of ``total`` proportional to ``shares`` (kept for tests)."""
    raw = shares * total
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    if rem > 0:
        base[np.argsort(raw - base)[::-1][:rem]] += 1
    return base


class PolicingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config | None = None, *, eval_mode: bool = False):
        super().__init__()
        self.cfg = cfg or Config()
        self.eval_mode = eval_mode

        self.region = region_for(self.cfg)
        self.R = self.region.n
        self.pop_weight = self.region.pop_weight
        self.uniform_share = np.full(self.R, 1.0 / self.R)
        self.gen = CrimeGenerator(self.cfg.abm, self.cfg.detection, self.region)
        self.fair = FairnessMeter(self.cfg.fairness, self.region)

        self._z_hardship = _z(self.region.hardship)
        self._socio = np.stack([
            self.region.hardship, self.region.poverty,
            _z(self.region.income), _z(self.region.unemployment),
        ], axis=1).astype(np.float32)

        self._ewma_decay = 0.5 ** (1.0 / EWMA_HALFLIFE)
        self._rule = self.cfg.env.action_mode == "rule"

        h = self.cfg.env.history
        obs_dim = h * self.R + self.R + 3
        if self.cfg.env.include_socioeconomic_obs:
            obs_dim += self._socio.size
        if self.cfg.env.observe_latent:
            obs_dim += self.R
        self.observation_space = spaces.Box(-1e6, 1e6, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0,
                                       (3,) if self._rule else (self.R,), np.float32)

        self._logit_temp = 8.0
        self._reward_scale = 200.0
        self._rng = np.random.default_rng(self.cfg.env.seed)
        self._episode_seed = self.cfg.env.seed

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        parts = [self._hist.reshape(-1) / 50.0,
                 self._last_deploy / max(self.cfg.env.n_units, 1),
                 [np.sin(2 * np.pi * self._hour / 24),
                  np.cos(2 * np.pi * self._hour / 24),
                  self._t / self.cfg.env.shifts_per_episode]]
        if self.cfg.env.include_socioeconomic_obs:
            parts.append(self._socio.reshape(-1))
        if self.cfg.env.observe_latent:
            parts.append(self._ewma_lat / 50.0)
        return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = seed
            if not self.eval_mode:
                # each training env draws later episode seeds from its own stream; with
                # one stream per cfg.env.seed, all parallel envs replayed the same episodes
                self._rng = np.random.default_rng([self.cfg.env.seed, seed])
        elif not self.eval_mode:
            self._episode_seed = int(self._rng.integers(1 << 31))
        g_rng = np.random.default_rng(self._episode_seed)
        self.gen.reset(g_rng)
        self.gen._rng = g_rng
        self._det_rng = np.random.default_rng(self._episode_seed + 7)
        self.fair.reset()

        self._t = 0
        self._hist = np.zeros((self.cfg.env.history, self.R), np.float32)
        self._last_deploy = self.cfg.env.n_units * self.pop_weight
        self._last_latent = np.zeros(self.R)
        self._ewma_rec = np.ones(self.R)
        self._ewma_lat = np.ones(self.R)
        self._hour = 0

        for _ in range(max(self.cfg.env.warmup_shifts, self.cfg.env.history)):
            self._advance(self._last_deploy.copy())
        self.fair.reset()
        self._t = 0
        return self._obs(), {}

    # ------------------------------------------------------------------
    def _advance(self, deployment: np.ndarray) -> dict:
        out = self.gen.run_shift(deployment)
        intensity = out["intensity_region"]
        rec = record(out["latent_region"], intensity, self.cfg.detection, self._det_rng)

        d = self._ewma_decay
        self._ewma_rec = d * self._ewma_rec + (1 - d) * rec["recorded"]
        self._ewma_lat = d * self._ewma_lat + (1 - d) * out["latent_region"]
        self._hist = np.roll(self._hist, -1, axis=0)
        self._hist[-1] = rec["recorded"]
        self._last_deploy = deployment
        self._last_latent = out["latent_region"]
        self._hour = out["hour"]

        return {
            "latent_region": out["latent_region"], "latent_cell": out["latent_cell"],
            "recorded_region": rec["recorded"], "intensity_region": intensity,
            "deployment": deployment,
            "latent_total": float(out["latent_region"].sum()),
            "recorded_total": float(rec["recorded"].sum()),
        }

    # ------------------------------------------------------------------
    def rule_params(self, action: np.ndarray) -> tuple[float, float, float]:
        """(beta, w, mu) of the allocation rule encoded by a rule-mode action.

        ``env.w_mode`` restricts the hardship tilt (JASSS A11): "free" w in [-2, 2],
        "zero" w = 0 (the policy acts through beta and mu only), "nonpos" w in [-2, 0].
        """
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        mode = self.cfg.env.w_mode
        if mode == "free":
            w = 2.0 * a[1]
        elif mode == "zero":
            w = 0.0
        elif mode == "nonpos":
            w = -(a[1] + 1.0)
        else:
            raise ValueError(f"unknown w_mode={mode!r}")
        return 2.0 * (a[0] + 1.0), w, 0.5 * (a[2] + 1.0)        # [0,4], w, [0,1]

    def action_to_deployment(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        if self._rule:
            beta, w, mu = self.rule_params(a)
            sig = self._ewma_rec / max(self._ewma_rec.mean(), 1e-9)
            sig = np.power(np.clip(sig, 1e-6, None), beta) * np.exp(w * self._z_hardship)
            shares = sig / sig.sum()
            shares = (1.0 - mu) * shares + mu * self.uniform_share
        else:
            shares = _softmax(a * self._logit_temp)
        f = self.cfg.env.min_share
        if f > 0:
            shares = f + (1.0 - f * self.R) * shares
        return self.cfg.env.n_units * shares / shares.sum()

    def step(self, action: np.ndarray):
        return self.step_with_deployment(self.action_to_deployment(action))

    def step_with_deployment(self, deployment: np.ndarray):
        step_info = self._advance(deployment)
        fair_info = self.fair.update(deployment)

        rec, lat = step_info["recorded_region"], step_info["latent_region"]
        dep_share = deployment / max(deployment.sum(), 1e-9)
        rec_cov = float((dep_share * rec).sum() / max(rec.sum(), 1.0))
        lat_cov = float((dep_share * lat).sum() / max(lat.sum(), 1.0))

        mode = self.cfg.env.reward_on
        sc = self.cfg.env.reward_scale_coverage
        if mode in ("captured", "recorded_coverage"):
            reward = sc * rec_cov
        elif mode == "latent_coverage":
            reward = sc * lat_cov
        elif mode == "latent":
            reward = -step_info["latent_total"] / self._reward_scale
        elif mode == "recorded_neg":
            reward = -step_info["recorded_total"] / self._reward_scale
        else:
            raise ValueError(f"unknown reward_on={mode!r}")

        self._t += 1
        truncated = self._t >= self.cfg.env.shifts_per_episode
        info = {**step_info, **fair_info,
                "recorded_coverage": rec_cov, "latent_coverage": lat_cov}
        return self._obs(), float(reward), False, truncated, info


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)
