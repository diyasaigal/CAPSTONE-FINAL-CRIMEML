import numpy as np
import pytest

from crimesim.abm import dynamics as dyn


def test_ring_sum_matches_bruteforce():
    rng = np.random.default_rng(0)
    a = rng.random((7, 5))
    out = dyn.ring_sum(a)
    G0, G1 = a.shape
    brute = np.zeros_like(a)
    for i in range(G0):
        for j in range(G1):
            for dx, dy in dyn.MOORE:
                brute[i, j] += a[(i + dx) % G0, (j + dy) % G1]
    assert np.allclose(out, brute)


def test_ring_sum_conserves_mass():
    a = np.random.default_rng(1).random((10, 10))
    # every cell contributes to exactly 8 neighbours
    assert np.isclose(dyn.ring_sum(a).sum(), 8 * a.sum())


def test_moore_choice_moves_toward_target():
    ax = np.array([5, 5, 0])
    ay = np.array([5, 5, 0])
    tx = np.array([9, 5, 0])   # far east / already there / already there
    ty = np.array([5, 9, 0])
    nx, ny = dyn.moore_choice_towards(ax, ay, tx, ty, grid=10)
    # agent 0 should step east (x increases), toward (9,5)
    assert nx[0] == 6 and ny[0] == 5
    # agent 1 should step north (y increases) toward (5,9)
    assert nx[1] == 5 and ny[1] == 6
    # always lands on a Moore neighbour (never stays put -- matches v5)
    for k in range(3):
        assert (abs(int(nx[k]) - int(ax[k])) in (1, 9)) or (abs(int(ny[k]) - int(ay[k])) in (1, 9))


def test_mean_abs_dev_matches_definition():
    rng = np.random.default_rng(2)
    pop = rng.normal(size=200)
    vals = rng.normal(size=37)
    got = dyn.mean_abs_dev(vals, pop)
    want = np.array([np.mean(np.abs(v - pop)) for v in vals])
    assert np.allclose(got, want, atol=1e-12)


def test_survival_matches_definition():
    rng = np.random.default_rng(3)
    pop = rng.random(500)
    vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    got = dyn.survival(vals, pop)
    want = np.array([(pop > v).mean() for v in vals])
    assert np.allclose(got, want)


def test_crime_events_shapes_and_monotone_in_risk():
    from crimesim.config import ABMConfig
    cfg = ABMConfig()
    n = 50
    rng = np.random.default_rng(4)
    G = cfg.grid
    common = dict(
        hardship_ca=np.full(77, 0.5), ca=np.zeros(n, int),
        ax=np.zeros(n, int), ay=np.zeros(n, int),
        cell_n=np.zeros((G, G)), cell_emp=np.zeros((G, G)),
        employed=np.ones(n, bool), hour=12, cfg=cfg,
    )
    low = dyn.crime_events(np.random.default_rng(4), stress=np.full(n, 0.6),
                           risk_grid=np.full((G, G), 0.1), past_risk=np.zeros(n), **common)
    high = dyn.crime_events(np.random.default_rng(4), stress=np.full(n, 0.6),
                            risk_grid=np.full((G, G), 10.0), past_risk=np.zeros(n), **common)
    assert low[0].shape == (n,)
    assert high[0].sum() > low[0].sum()          # more environmental risk -> more crime
    assert set(np.unique(high[0])).issubset({0, 1, 2})
