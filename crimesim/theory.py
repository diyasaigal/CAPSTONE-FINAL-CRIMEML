"""Mean-field model of the allocation rule under patrol-dependent recording.

Districts i = 1..R with untreated crime rate ``lam_i`` and population weight
``pi_i``. A patrol share ``s_i`` gives intensity ``u_i = s_i * B / (B_ref * pi_i)``.
Patrol acts on the district twice (``DetectionConfig``):

    latent crime     L_i = lam_i * (1 - delta(u_i)),  delta(u) = min(cap, kappa * u)
    recorded crime   R_i = L_i * q(u_i),              q(u) = p0 + g * min(1, u / u_sat)

The rule of ``PolicingEnv.action_to_deployment`` allocates

    s_i  proportional to  (x_i / mean x)^beta * exp(w * z_i),   then mixes with uniform (mu)
                                                                 and applies the floor f,

where the signal x is recorded crime (deployable) or latent crime (oracle). A
steady state is a fixed point of that map. Writing h(u) for the factor by which
patrol scales the signal (h = q * (1 - delta) for recorded, 1 - delta for
latent) and eps(u) = d log h / d log u for its elasticity, linearising around
a state with common elasticity eps gives

    log s_i = (beta * log lam_i + w * z_i) / (1 - beta * eps) + const,

so the feedback multiplies whatever tilt the rule intends by 1 / (1 - beta * eps)
and the fixed point is locally stable only while beta * eps < 1. With
patrol-independent recording (g = 0) eps = -kappa u / (1 - kappa u) <= 0: the
loop damps concentration instead of amplifying it.
"""
from __future__ import annotations

import numpy as np

from crimesim.config import DetectionConfig


def q(u, det: DetectionConfig):
    return det.p_report + det.detect_gain * np.clip(u / det.detect_saturation, 0.0, 1.0)


def keep(u, det: DetectionConfig):
    return 1.0 - np.minimum(det.deterrence_cap, det.deterrence_slope * u)


def signal_factor(u, det: DetectionConfig, signal: str = "recorded"):
    """h(u): how patrol intensity scales the signal the rule reads."""
    return keep(u, det) * (q(u, det) if signal == "recorded" else 1.0)


def elasticity(u, det: DetectionConfig, signal: str = "recorded") -> float:
    """eps(u) = d log h / d log u (analytic, piecewise)."""
    e = 0.0
    if det.deterrence_slope * u < det.deterrence_cap:
        e -= det.deterrence_slope * u / (1.0 - det.deterrence_slope * u)
    if signal == "recorded" and u < det.detect_saturation:
        e += (det.detect_gain * u / det.detect_saturation) / q(u, det)
    return float(e)


def rule_shares(x, beta, w, z, mu=0.0, floor=0.005):
    x = np.asarray(x, float)
    sig = np.power(np.clip(x / max(x.mean(), 1e-12), 1e-6, None), beta) * np.exp(w * z)
    s = sig / sig.sum()
    s = (1.0 - mu) * s + mu / len(s)
    if floor > 0:
        s = floor + (1.0 - floor * len(s)) * s
    return s / s.sum()


def fixed_point(lam, det: DetectionConfig, *, beta=1.0, w=0.0, z=None, mu=0.0, floor=0.005,
                pop=None, budget_ratio=1.0, signal="recorded", damping=0.2, iters=5000,
                tol=1e-12) -> dict:
    """Steady-state shares of the rule, by damped iteration from uniform.

    ``damping`` plays the role of the EWMA: s <- (1 - damping) * s + damping * rule(x(s)).
    Returns shares, intensities and whether the iteration converged.
    """
    lam = np.asarray(lam, float)
    R = len(lam)
    z = np.zeros(R) if z is None else np.asarray(z, float)
    pop = np.full(R, 1.0 / R) if pop is None else np.asarray(pop, float)
    s = np.full(R, 1.0 / R)
    ok = False
    for _ in range(iters):
        u = budget_ratio * s / pop
        new = rule_shares(lam * signal_factor(u, det, signal), beta, w, z, mu, floor)
        nxt = (1.0 - damping) * s + damping * new
        if np.abs(nxt - s).max() < tol:
            s, ok = nxt, True
            break
        s = nxt
    u = budget_ratio * s / pop
    return {"shares": s, "intensity": u, "converged": ok,
            "latent": lam * keep(u, det), "recorded": lam * keep(u, det) * q(u, det)}


def group_disparity(shares, groups, pop=None) -> float:
    """Exposure of the top group / bottom group (``FairnessMeter`` definition)."""
    shares = np.asarray(shares, float)
    pop = np.full(len(shares), 1.0 / len(shares)) if pop is None else np.asarray(pop, float)
    hi, lo = groups == groups.max(), groups == groups.min()
    return float((shares[hi].sum() / pop[hi].sum()) / (shares[lo].sum() / pop[lo].sum()))


def coverage(shares, crime) -> float:
    """Patrol-weighted share of ``crime`` (the coverage reward)."""
    return float((shares * crime).sum() / crime.sum())
