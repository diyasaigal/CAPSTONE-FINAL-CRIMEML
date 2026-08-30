"""Pure, vectorised helper functions for the emergent-crime generator.

These implement a mean-field version of the per-agent rules in
``legacy/crime_model_v5.py``. Because v5 assigns each agent a *random* community
area independent of its grid position, the neighbour-interaction terms in v5 are
already effectively mean-field; here we make that explicit so the whole model is
array-vectorised and fast enough for RL (millions of steps).

Every function is deterministic given its inputs (RNG passed in explicitly) and
is unit-tested in ``tests/test_dynamics.py``.
"""
from __future__ import annotations

import numpy as np

# 8 Moore-neighbourhood offsets (no centre), matching v5's get_neighborhood.
MOORE = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)])


def ring_sum(a: np.ndarray) -> np.ndarray:
    """Toroidal sum of ``a`` over the 8 Moore neighbours of each cell.

    One wrap-pad + 8 slice adds -- ~5x faster than 8 ``np.roll`` calls.
    Works on a stack: if ``a`` is (K, G, G) each layer is summed independently.
    """
    if a.ndim == 3:
        p = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="wrap")
        return (p[:, :-2, :-2] + p[:, :-2, 1:-1] + p[:, :-2, 2:]
                + p[:, 1:-1, :-2] + p[:, 1:-1, 2:]
                + p[:, 2:, :-2] + p[:, 2:, 1:-1] + p[:, 2:, 2:])
    p = np.pad(a, 1, mode="wrap")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])


def moore_choice_towards(ax, ay, tx, ty, grid):
    """For each agent pick the toroidal Moore neighbour closest to its target.

    Distance is measured on the *wrapped* coordinates, matching v5 which feeds the
    coordinates returned by Mesa's toroidal ``get_neighborhood`` straight into a
    squared-distance comparison.
    """
    cand_x = (ax[:, None] + MOORE[:, 0][None, :]) % grid      # (N, 8)
    cand_y = (ay[:, None] + MOORE[:, 1][None, :]) % grid
    d2 = (cand_x - tx[:, None]) ** 2 + (cand_y - ty[:, None]) ** 2
    k = d2.argmin(axis=1)
    idx = np.arange(len(ax))
    return cand_x[idx, k], cand_y[idx, k]


def _sorted_prefix(pop: np.ndarray):
    s = np.sort(pop)
    prefix = np.concatenate([[0.0], np.cumsum(s)])
    return s, prefix


def mean_abs_dev(values: np.ndarray, pop: np.ndarray, _cache=None) -> np.ndarray:
    """E_{p ~ pop} |v - p| for every v in ``values`` (vectorised, O((N+M) log M)).

    ``_cache`` may be a ``(sorted_pop, prefix_sums)`` tuple from :func:`_sorted_prefix`
    to skip the sort when the same population is reused within a step.
    """
    s, prefix = _cache if _cache is not None else _sorted_prefix(pop)
    total = prefix[-1]
    n = len(s)
    c = np.searchsorted(s, values, side="right")
    sum_below = prefix[c]
    below = values * c - sum_below
    above = (total - sum_below) - values * (n - c)
    return (below + above) / n


def survival(values: np.ndarray, pop: np.ndarray, _cache=None) -> np.ndarray:
    """P_{p ~ pop}(p > v) for every v in ``values``."""
    s = _cache[0] if _cache is not None else np.sort(pop)
    return (len(s) - np.searchsorted(s, values, side="right")) / len(s)


def escalation_term(stress, employed, nbr_n, nbr_emp, gamma, _cache):
    """Mean-field interaction-escalation I for each agent.

    v5 per neighbour: ``0.1 * (employment differs) + 0.2 * |stress_i - stress_n|``.
    Mean-field: employment-mismatch uses neighbour employed/unemployed counts;
    the stress term uses the population mean-absolute-deviation of the agent's
    stress scaled by the neighbour count.
    """
    emp_mismatch = 0.1 * np.where(employed, nbr_n - nbr_emp, nbr_emp)
    stress_term = 0.2 * nbr_n * mean_abs_dev(stress, stress, _cache)
    return gamma * (emp_mismatch + stress_term)


def crime_events(rng, *, stress, hardship_ca, ca, ax, ay, risk_grid, past_risk,
                 cell_n, cell_emp, employed, hour, cfg):
    """One vectorised application of v5's two crime-trigger paths.

    Returns ``(n_events_per_agent, new_past_risk, new_stress)`` where
    ``n_events_per_agent`` is 0/1/2 (an agent can trip both paths in a step, as in v5).
    """
    cache = _sorted_prefix(stress)                # one sort, reused below
    nbr_n = cell_n[ax, ay]
    nbr_emp = cell_emp[ax, ay]
    S = stress + hardship_ca[ca]
    T = risk_grid[ax, ay]
    I = escalation_term(stress, employed, nbr_n, nbr_emp, cfg.gamma, cache)

    night = (hour >= 20) or (hour <= 2)
    time_factor = cfg.night_factor if night else 1.0

    current = (S * T + I) * time_factor
    final = (1.0 - cfg.risk_memory) * current + cfg.risk_memory * past_risk
    new_past_risk = final.copy()

    # --- interaction-driven escalation (v5: mean pairwise stress > trigger) ---
    thresh = 2.0 * cfg.interaction_trigger - stress          # stress_n needed
    k = nbr_n * survival(thresh, stress, cache)              # expected qualifying neighbours
    p_int = 1.0 - (1.0 - cfg.interaction_prob) ** k
    crime_int = rng.random(len(stress)) < p_int

    # --- environmental threshold (v5: final risk > trigger) ---
    crime_thr = final > cfg.risk_trigger

    new_stress = np.where(crime_thr, stress * cfg.crime_stress_relief, stress)
    n_events = crime_int.astype(np.int64) + crime_thr.astype(np.int64)
    return n_events, new_past_risk, new_stress
