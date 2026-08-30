"""Rule-based patrol-allocation policies used as comparisons for the RL agent.

Every policy exposes ``reset()`` and ``act(env) -> deployment``, a length-R
fractional vector summing to the patrol budget. They read only what a real
deployed system could see (``env._hist`` = recorded crime history, static
covariates); ``GreedyOracle`` additionally peeks at latent crime and is an upper
bound on efficiency, not a deployable policy.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _alloc(scores: np.ndarray, env) -> np.ndarray:
    """Fractional deployment of the patrol budget proportional to ``scores``,
    with the same per-district floor the RL action space enforces."""
    scores = np.clip(np.asarray(scores, float), 0, None)
    if scores.sum() <= 0:
        scores = np.ones_like(scores)
    shares = scores / scores.sum()
    f = env.cfg.env.min_share
    if f > 0:
        shares = f + (1.0 - f * len(shares)) * shares
    return env.cfg.env.n_units * shares


class UniformPolicy:
    """Allocate proportional to area size (share of real crime). Fair by construction."""
    name = "uniform"

    def reset(self): pass

    def act(self, env):
        return _alloc(env.pop_weight.copy(), env)


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self): pass

    def act(self, env):
        return _alloc(self.rng.random(env.R), env)


class EWMAHotspotPolicy:
    """PredPol-style: allocate proportional to an exponentially-weighted moving
    average of *recorded* crime. This is the canonical predictive-policing
    heuristic and the one Ensign et al. (2018) show is prone to runaway loops."""
    name = "ewma_hotspot"

    def __init__(self, halflife: float = 5.0):
        self.decay = 0.5 ** (1.0 / halflife)
        self.ewma = None

    def reset(self):
        self.ewma = None

    def act(self, env):
        recent = env._hist[-1].astype(float)
        self.ewma = recent if self.ewma is None else self.decay * self.ewma + (1 - self.decay) * recent
        return _alloc(self.ewma.copy(), env)


class RandomForestHotspotPolicy:
    """The legacy ``ml_validation`` model, made sequential: a RandomForest predicts
    next-shift recorded crime per CA from static covariates + recent recorded
    history, refit every ``refit_every`` shifts on accumulated observations."""
    name = "rf_hotspot"

    def __init__(self, refit_every: int = 15, seed: int = 0):
        self.refit_every = refit_every
        self.seed = seed
        self.reset()

    def reset(self):
        self.model = None
        self.X, self.y = [], []
        self._prev_feat = None
        self._k = 0

    def _features(self, env):
        inc = env.region.income
        return np.concatenate([
            env._hist.reshape(-1),
            env.region.hardship, env.region.poverty,
            (inc - inc.mean()) / (inc.std() + 1e-9),
        ])

    def act(self, env):
        feat = self._features(env)
        if self._prev_feat is not None:
            self.X.append(self._prev_feat)
            self.y.append(env._hist[-1].astype(float))
        self._prev_feat = feat
        self._k += 1

        enough = len(self.X) > 8
        due = self.model is None or self._k % self.refit_every == 0
        if enough and due:
            m = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=self.seed)
            m.fit(np.array(self.X), np.array(self.y))
            self.model = m

        if self.model is None:
            return _alloc(env._hist[-1].astype(float) + 1e-3, env)
        pred = self.model.predict(feat[None])[0]
        return _alloc(pred, env)


class GreedyOracle:
    """Allocate proportional to an EWMA of *latent* crime. Not deployable -- an
    upper bound on how much a well-targeted policy can reduce real crime."""
    name = "oracle"

    def __init__(self, halflife: float = 5.0):
        self.decay = 0.5 ** (1.0 / halflife)
        self.ewma = None

    def reset(self):
        self.ewma = None

    def act(self, env):
        cur = env._last_latent.astype(float)
        self.ewma = cur if self.ewma is None else self.decay * self.ewma + (1 - self.decay) * cur
        base = self.ewma if self.ewma.sum() > 0 else env.pop_weight
        return _alloc(base, env)


BASELINES = {
    p.name: p for p in [
        UniformPolicy(), RandomPolicy(), EWMAHotspotPolicy(),
        RandomForestHotspotPolicy(), GreedyOracle(),
    ]
}


def make_baselines() -> dict:
    return {
        "uniform": UniformPolicy(),
        "random": RandomPolicy(),
        "ewma_hotspot": EWMAHotspotPolicy(),
        "rf_hotspot": RandomForestHotspotPolicy(),
        "oracle": GreedyOracle(),
    }
