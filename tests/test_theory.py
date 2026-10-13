import numpy as np
import pytest

from crimesim import theory as th
from crimesim.config import DetectionConfig

DET = DetectionConfig()                                   # p0 .30, gain .55, sat 1.5, kappa .26, cap .68
CONST = DetectionConfig(p_report=0.667, detect_gain=0.0)
LAM = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
GROUPS = np.array([0, 0, 1, 1, 2, 2])


def test_elasticity_matches_closed_form_and_numeric_derivative():
    assert th.elasticity(1.0, DET) == pytest.approx(0.55 / 1.5 / (0.30 + 0.55 / 1.5) - 0.26 / 0.74)
    assert th.elasticity(1.0, DET, "latent") == pytest.approx(-0.26 / 0.74)
    for u in (0.4, 1.0, 1.4, 2.0):
        h = 1e-6
        num = (np.log(th.signal_factor(u + h, DET)) - np.log(th.signal_factor(u - h, DET))) / (2 * h) * u
        assert th.elasticity(u, DET) == pytest.approx(num, abs=1e-5)


def test_recording_loop_amplifies_only_when_detection_depends_on_patrol():
    assert th.elasticity(1.0, DET) > 0                    # patrol raises what is recorded
    assert th.elasticity(1.0, CONST) < 0                  # only deterrence left: damping
    assert th.elasticity(2.0, DET) < 0                    # past detection saturation
    assert th.elasticity(3.0, DET) == 0                   # past the deterrence cap as well


def test_beta_zero_is_uniform_and_fixed_point_is_self_consistent():
    fp = th.fixed_point(LAM, DET, beta=0.0, floor=0.0)
    assert np.allclose(fp["shares"], 1 / len(LAM))
    fp = th.fixed_point(LAM, DET, beta=2.0, floor=0.005)
    assert fp["converged"]
    again = th.rule_shares(fp["recorded"], 2.0, 0.0, np.zeros(len(LAM)), 0.0, 0.005)
    assert np.allclose(again, fp["shares"], atol=1e-8)
    assert fp["shares"].sum() == pytest.approx(1.0)


def test_linearised_slope_matches_fixed_point_for_small_tilts():
    # near-equal districts keep every u close to 1, where the constant-elasticity formula holds
    lam = 1.0 + 0.02 * np.arange(6)
    for det in (DET, CONST):
        eps = th.elasticity(1.0, det)
        for beta in (1.0, 2.0):
            s = th.fixed_point(lam, det, beta=beta, floor=0.0)["shares"]
            slope = np.polyfit(np.log(lam), np.log(s), 1)[0]
            assert slope == pytest.approx(beta / (1 - beta * eps), rel=0.02)


def test_contaminated_signal_concentrates_more_than_constant_detection():
    d = lambda det, sig="recorded": th.group_disparity(
        th.fixed_point(LAM, det, beta=2.0, signal=sig)["shares"], GROUPS)
    assert d(DET) > d(CONST) > 1
    assert d(CONST) == pytest.approx(d(DET, "latent"))    # constant q cancels in the shares


def test_group_disparity_and_coverage():
    s = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    assert th.group_disparity(s, GROUPS) == pytest.approx(2.0)
    assert th.coverage(s, np.array([0, 0, 0, 0, 1.0, 1.0])) == pytest.approx(0.2)
