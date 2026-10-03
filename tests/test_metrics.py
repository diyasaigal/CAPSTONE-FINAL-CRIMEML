from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crimesim.abm.regions import load_region_info
from crimesim.config import load_config
from crimesim.eval.incidence import need_controlled_incidence
from crimesim.eval.oracle_relative import (add_reference_shift, episode_ids, episode_metrics,
                                           episode_table, reference_disparity, relative_summary)
from crimesim.eval.rollout import rollout
from crimesim.eval.stats import hierarchical_bootstrap, ols
from crimesim.rl.baselines import EWMAHotspotPolicy, GreedyOracle


def _frame(dep, lat, rec=None, seed=0):
    dep, lat = np.asarray(dep, float), np.asarray(lat, float)
    rec = 0.5 * lat if rec is None else np.asarray(rec, float)
    T = len(dep)
    return pd.DataFrame({"shift": np.arange(T), "seed": seed,
                         "deployment": list(dep), "latent_region": list(lat),
                         "recorded_region": list(rec),
                         "latent_coverage": 0.05, "recorded_coverage": 0.06})


# ---------------------------------------------------------------- episodes
def test_episode_ids_split_on_shift_reset_even_with_repeated_seed():
    df = pd.DataFrame({"shift": [0, 1, 2, 0, 1, 0], "seed": [7] * 6})
    assert episode_ids(df).tolist() == [1, 1, 1, 2, 2, 3]


def test_episode_disparity_matches_fairness_meter():
    cfg = load_config(env={"shifts_per_episode": 12, "warmup_shifts": 2})
    region = load_region_info(cfg.abm.grid)
    df = rollout(EWMAHotspotPolicy(), cfg, seeds=[1, 2], keep_arrays=True)
    t = episode_table(df, region)
    logged = df.groupby(episode_ids(df))["disparity_cum"].last().to_numpy()
    assert len(t) == 2
    assert np.allclose(t["disparity"], logged)


# ---------------------------------------------------------------- episode metrics
def test_uniform_patrol_has_unit_disparity_and_undistorted_records():
    R = 10
    groups = np.repeat(np.arange(5), 2)
    lat = np.tile(np.arange(1, R + 1), (4, 1))
    m = episode_metrics(_frame(np.ones((4, R)), lat), groups, np.full(R, 1 / R))
    assert m["disparity"] == pytest.approx(1.0)
    assert m["rec_distortion"] == pytest.approx(1.0)       # recorded is proportional to latent
    ls = np.arange(1, R + 1) / np.arange(1, R + 1).sum()
    assert m["need_tv"] == pytest.approx(0.5 * np.abs(1 / R - ls).sum())
    assert m["proxy_gap"] == pytest.approx(0.01)


def test_need_proportional_patrol_has_zero_need_distance():
    R = 10
    lat = np.tile(np.arange(1, R + 1), (3, 1))
    m = episode_metrics(_frame(lat, lat), np.repeat(np.arange(5), 2), np.full(R, 1 / R))
    assert m["need_tv"] == pytest.approx(0.0)
    assert m["disparity"] == pytest.approx((9 + 10) / (1 + 2))


def test_over_recording_in_patrolled_quantile_shows_as_distortion():
    R = 10
    groups = np.repeat(np.arange(5), 2)
    lat = np.ones((2, R))
    rec = lat.copy()
    rec[:, groups == 4] *= 2                                # top quantile recorded twice as often
    m = episode_metrics(_frame(np.ones((2, R)), lat, rec), groups, np.full(R, 1 / R))
    assert m["rec_distortion"] == pytest.approx(2.0)


# ---------------------------------------------------------------- oracle-relative
def _tables():
    ref = pd.DataFrame({"seed": [1, 2, 3], "disparity": [2.0, 3.0, 4.0], "tseed": -1})
    pol = pd.DataFrame({"seed": [1, 2, 3] * 2, "tseed": [0] * 3 + [1] * 3,
                        "disparity": [6.0, 9.0, 12.0, 4.0, 6.0, 8.0]})
    for t in (ref, pol):
        for c in ["need_tv", "lat_cov", "rec_cov", "proxy_gap", "rec_distortion"]:
            t[c] = 0.0
    return ref, pol


def test_reference_shift_is_log_ratio_on_same_seed():
    ref, pol = _tables()
    r = reference_disparity(ref)
    out = add_reference_shift(pol, r)
    assert np.allclose(out["dlog"], np.log([3, 3, 3, 2, 2, 2]))
    assert np.allclose(add_reference_shift(ref, r)["dlog"], 0.0)


def test_reference_shift_refuses_unmatched_seeds():
    ref, pol = _tables()
    with pytest.raises(ValueError, match="eval seeds"):
        add_reference_shift(pol.assign(seed=[1, 2, 9] * 2), reference_disparity(ref))


def test_relative_summary_reports_geometric_ratio():
    ref, pol = _tables()
    r = reference_disparity(ref)
    s = relative_summary(add_reference_shift(pol.assign(family="p"), r), ["family"], r, n_boot=500)
    row = s.iloc[0]
    assert row["x_ref"] == pytest.approx(np.sqrt(6.0))      # exp(mean(log 3, log 2))
    assert row["x_ref_lo"] <= row["x_ref"] <= row["x_ref_hi"]
    assert row["ratio_of_means"] == pytest.approx(7.5 / 3.0)
    assert row["n_train_seeds"] == 2 and row["n_eval_seeds"] == 3


# ---------------------------------------------------------------- statistics
def test_hierarchical_bootstrap_basic_properties():
    assert hierarchical_bootstrap(np.full((3, 5), 2.5)) == pytest.approx((2.5, 2.5, 2.5))
    M = np.random.default_rng(0).normal(size=(4, 20))
    m, lo, hi = hierarchical_bootstrap(M, n_boot=1000)
    assert lo < m < hi and m == pytest.approx(M.mean())
    assert hierarchical_bootstrap(M, n_boot=200, seed=3) == hierarchical_bootstrap(M, n_boot=200, seed=3)


def test_hierarchical_bootstrap_widens_with_training_seed_spread():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.1, (4, 20))
    tight = hierarchical_bootstrap(noise, n_boot=1000)
    spread = hierarchical_bootstrap(noise + np.array([[-1], [1], [-1], [1]]), n_boot=1000)
    assert spread[2] - spread[1] > 3 * (tight[2] - tight[1])


def test_singleton_clusters_reproduce_hc1():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 1 + 2 * x + rng.normal(size=200) * (1 + np.abs(x))
    hc1 = ols(y, {"x": x}).get("x")
    cr1 = ols(y, {"x": x}, cluster=np.arange(200)).get("x")
    assert cr1["coef"] == pytest.approx(hc1["coef"])
    assert cr1["se"] == pytest.approx(hc1["se"])


def test_clustered_se_exceeds_hc1_when_errors_are_shared_within_cluster():
    rng = np.random.default_rng(0)
    G, n = 20, 50
    cl = np.repeat(np.arange(G), n)
    x = np.repeat(rng.normal(size=G), n) + 0.1 * rng.normal(size=G * n)
    y = x + np.repeat(rng.normal(size=G), n) + 0.1 * rng.normal(size=G * n)
    assert ols(y, {"x": x}, cluster=cl).get("x")["se"] > 3 * ols(y, {"x": x}).get("x")["se"]


# ---------------------------------------------------------------- E3
def _incidence_frame(dep_fn, n_ep=6, T=40, R=12, seed=0):
    rng = np.random.default_rng(seed)
    hardship = np.linspace(0, 1, R)
    need = (1 + 2 * hardship) / (1 + 2 * hardship).sum()
    region = SimpleNamespace(hardship=hardship, poverty=rng.random(R))
    frames = [_frame([dep_fn(need, hardship, rng) for _ in range(T)], np.ones((T, R)), seed=e)
              for e in range(n_ep)]
    return pd.concat(frames, ignore_index=True), region, need


def test_need_proportional_patrol_shows_no_hardship_incidence():
    df, region, need = _incidence_frame(lambda n, h, rng: n * rng.uniform(0.9, 1.1, len(n)))
    r = need_controlled_incidence(df, region, need)["hardship"]
    assert abs(r["t"]) < 3


def test_hardship_tilted_patrol_shows_positive_incidence():
    df, region, need = _incidence_frame(
        lambda n, h, rng: n * np.exp(0.8 * h) * rng.uniform(0.9, 1.1, len(n)))
    r = need_controlled_incidence(df, region, need)
    assert r["hardship"]["coef"] > 0 and r["hardship"]["p"] < 0.01
    assert r["n_clusters"] == 6
