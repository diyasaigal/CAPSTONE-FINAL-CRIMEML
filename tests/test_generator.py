import numpy as np
import pytest

from crimesim.abm.generator import CrimeGenerator
from crimesim.abm.regions import load_region_info
from crimesim.config import ABMConfig, DetectionConfig


@pytest.fixture(scope="module")
def region():
    return load_region_info(20)


def _gen(region, **abm):
    return CrimeGenerator(ABMConfig(**abm), DetectionConfig(), region)


def test_unseeded_start(region):
    g = _gen(region)
    g.reset(np.random.default_rng(0))
    assert np.allclose(g.risk_grid, 0.1)          # uniform, no crime coordinates injected


def test_determinism_same_seed(region):
    a = _gen(region); a.reset(np.random.default_rng(7))
    b = _gen(region); b.reset(np.random.default_rng(7))
    ta = [a.run_shift(None)["latent_region"].sum() for _ in range(20)]
    tb = [b.run_shift(None)["latent_region"].sum() for _ in range(20)]
    assert ta == tb


def test_different_seeds_differ(region):
    a = _gen(region); a.reset(np.random.default_rng(1))
    b = _gen(region); b.reset(np.random.default_rng(2))
    ta = sum(a.run_shift(None)["latent_region"].sum() for _ in range(20))
    tb = sum(b.run_shift(None)["latent_region"].sum() for _ in range(20))
    assert ta != tb


def test_latent_cell_and_region_consistent(region):
    g = _gen(region); g.reset(np.random.default_rng(3))
    out = g.run_shift(None)
    assert out["latent_cell"].shape == (20, 20)
    assert out["latent_region"].shape == (region.n,)
    assert np.isclose(out["latent_cell"].sum(), out["latent_region"].sum())
    assert (out["latent_cell"] >= 0).all()


def test_emergent_clustering(region):
    g = _gen(region); g.reset(np.random.default_rng(0))
    acc = np.zeros((20, 20))
    for _ in range(60):
        acc += g.run_shift(None)["latent_cell"]
    assert acc.sum() > 0
    assert acc.max() / acc.mean() > 3.0


def test_patrol_reduces_crime(region):
    det = DetectionConfig(deterrence_slope=0.3, deterrence_cap=0.6)
    def total(patrol):
        g = CrimeGenerator(ABMConfig(), det, region)
        g.reset(np.random.default_rng(0))
        return sum(g.run_shift(patrol)["latent_region"].sum() for _ in range(40))
    assert total(np.full(region.n, 5.0)) < total(None)


def test_patrol_intensity_normalisation(region):
    g = _gen(region); g.reset(np.random.default_rng(0))
    prop = g.pop_weight * g.cfg.patrol_reference
    assert np.allclose(g.patrol_intensity(prop), 1.0, atol=1e-6)
    assert np.allclose(g.patrol_intensity(prop * 2), 2.0, atol=1e-6)


def test_turnover_keeps_population_stationary(region):
    g = _gen(region, turnover_rate=0.05, stress_recovery=0.18)
    g.reset(np.random.default_rng(0))
    for _ in range(20):                       # burn in the transient
        g.run_shift(None)
    early = np.mean([g.run_shift(None)["latent_region"].sum() for _ in range(30)])
    for _ in range(60):
        g.run_shift(None)
    late = np.mean([g.run_shift(None)["latent_region"].sum() for _ in range(30)])
    assert 0.7 < late / early < 1.4


def test_sub_step_counts_add_up_to_the_shift(region):
    gen = CrimeGenerator(ABMConfig(), DetectionConfig(), region)
    gen.reset(np.random.default_rng(0))
    for _ in range(8):
        out = gen.run_shift(None)
        assert np.isclose(out["sub_counts"].sum(), out["latent_cell"].sum())
        assert out["sub_hours"][-1] == out["hour"] and len(out["sub_hours"]) == 4


def test_night_day_ratio_uses_hourly_window():
    from crimesim.abm.calibration import is_night, night_day_ratio
    hours = np.arange(24)
    assert is_night(hours).sum() == 7            # 20, 21, 22, 23, 0, 1, 2
    counts = np.where(is_night(hours), 3.0, 1.0)
    assert night_day_ratio(hours, counts) == (3.0, 1.0)


def test_hardship_once_and_spread(region):
    g = _gen(region, hardship_in_risk=False)
    assert (g._hardship_risk == 0).all() and np.allclose(g._hardship, region.hardship)
    flat = _gen(region, hardship_scale=0.0)
    assert np.allclose(flat._hardship, flat._hardship[0])        # no hardship differences
    mean = (flat.pop_weight * region.hardship).sum()
    assert np.isclose(flat._hardship[0], mean)


def test_population_weighting(region):
    from crimesim.abm.regions import with_population
    assert with_population(region, False) is region
    w = with_population(region, True).pop_weight
    assert np.isclose(w.sum(), 1.0) and w.max() / w.min() > 2
    assert np.isclose(region.population.sum(), 2_647_621)         # ACS 2023, all 77 areas


def test_jasss_variants_load():
    from crimesim.experiments.jasss import GENERATOR_VARIANTS, jasss_config
    base = jasss_config()
    assert base.abm.population_weighted and base.abm.night_interaction
    assert not base.abm.hardship_in_risk and base.env.warmup_shifts == 100
    assert base.detection.p_report == 0.30                          # inherited from base.yaml
    assert jasss_config("contrast_high").abm.hardship_in_risk
    assert jasss_config("agents_500", rl={"seed": 3}).abm.n_agents == 500
    assert set(GENERATOR_VARIANTS) >= {"base", "contrast_low", "contrast_high"}
