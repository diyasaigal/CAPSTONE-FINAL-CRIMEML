import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from crimesim.config import load_config
from crimesim.env.policing_env import PolicingEnv, largest_remainder


def _cfg(**env):
    return load_config(env={"shifts_per_episode": 12, "history": 3, **env},
                       abm={"steps_per_shift": 4})


def test_largest_remainder_sums_exactly():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = rng.random(77); s /= s.sum()
        a = largest_remainder(s, 20)
        assert a.sum() == 20 and (a >= 0).all()


def test_gymnasium_api_compliant():
    env = PolicingEnv(_cfg())
    check_env(env, skip_render_check=True)


def test_reset_and_step_shapes():
    env = PolicingEnv(_cfg())
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    assert env.observation_space.contains(obs)
    assert np.isfinite(r)
    for k in ["latent_total", "recorded_total", "cost", "disparity_cum", "deployment"]:
        assert k in info
    assert np.isclose(info["deployment"].sum(), env.cfg.env.n_units)


def test_episode_terminates_on_horizon():
    env = PolicingEnv(_cfg(shifts_per_episode=10))
    env.reset(seed=0)
    steps = 0
    trunc = False
    while not trunc:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        steps += 1
        assert steps <= 11
    assert steps == 10


def test_reward_on_modes():
    env = PolicingEnv(_cfg(reward_on="captured"))
    sc = env.cfg.env.reward_scale_coverage
    env.reset(seed=0)
    _, r, _, _, info = env.step(np.zeros(env.action_space.shape[0], np.float32))
    assert r >= 0 and np.isclose(r, sc * info["recorded_coverage"])

    env2 = PolicingEnv(_cfg(reward_on="latent_coverage"))
    env2.reset(seed=0)
    _, r2, _, _, info2 = env2.step(np.zeros(env2.action_space.shape[0], np.float32))
    assert np.isclose(r2, sc * info2["latent_coverage"])

    env3 = PolicingEnv(_cfg(reward_on="latent"))
    env3.reset(seed=0)
    _, r3, _, _, info3 = env3.step(np.zeros(env3.action_space.shape[0], np.float32))
    assert np.isclose(-r3 * env3._reward_scale, info3["latent_total"])


def test_determinism_same_seed():
    e1 = PolicingEnv(_cfg()); e1.reset(seed=42)
    e2 = PolicingEnv(_cfg()); e2.reset(seed=42)
    a = np.zeros(e1.action_space.shape[0], np.float32)
    for _ in range(8):
        o1, r1, *_ = e1.step(a)
        o2, r2, *_ = e2.step(a)
        assert np.allclose(o1, o2) and r1 == r2
