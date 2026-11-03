"""Interrupted training continues from its last resume point (crimesim.rl.ppo)."""
import json

import numpy as np
import pytest

from crimesim.config import load_config
from crimesim.rl import ppo as P


class Interrupted(Exception):
    pass


def _cfg(**rl):
    return load_config(env={"shifts_per_episode": 20, "warmup_shifts": 2, "history": 2},
                       abm={"steps_per_shift": 1},
                       rl={"n_envs": 2, "n_steps": 256, "batch_size": 64, "total_timesteps": 1536,
                           "resume_every": 256, "seed": 0, **rl})


@pytest.fixture
def interrupt_after(monkeypatch):
    """Make training stop with an exception once it passes ``steps``."""
    def arm(steps):
        orig = P.ResumeCallback._on_step

        def _on_step(self):
            if self.num_timesteps > steps:
                raise Interrupted
            return orig(self)
        monkeypatch.setattr(P.ResumeCallback, "_on_step", _on_step)
        return lambda: monkeypatch.setattr(P.ResumeCallback, "_on_step", orig)
    return arm


def test_ppo_resumes_after_interruption(tmp_path, interrupt_after):
    cfg = _cfg()
    disarm = interrupt_after(900)
    with pytest.raises(Interrupted):
        P.train_ppo(cfg, save_dir=tmp_path)
    state = json.loads((P.resume_point(tmp_path) / "state.json").read_text())
    assert 512 <= state["num_timesteps"] <= 900
    assert not (tmp_path / "ppo_model.zip").exists()

    disarm()
    model, _ = P.train_ppo(cfg, save_dir=tmp_path)
    assert model.num_timesteps == cfg.rl.total_timesteps
    assert json.loads((tmp_path / "resumed_at.json").read_text()) == [state["num_timesteps"]]
    assert (tmp_path / "ppo_model.zip").exists() and P.resume_point(tmp_path) is None


def test_resume_refuses_a_changed_config(tmp_path, interrupt_after):
    interrupt_after(600)
    with pytest.raises(Interrupted):
        P.train_ppo(_cfg(), save_dir=tmp_path)
    with pytest.raises(RuntimeError, match="another config"):
        P.train_ppo(_cfg(learning_rate=1e-3), save_dir=tmp_path)


def test_lagrangian_multiplier_survives_resume(tmp_path, interrupt_after):
    from crimesim.rl.constrained_ppo import train_lagrangian_ppo
    cfg = load_config(env={"shifts_per_episode": 20, "warmup_shifts": 2, "history": 2},
                      abm={"steps_per_shift": 1},
                      rl={"n_envs": 2, "n_steps": 256, "batch_size": 64, "total_timesteps": 1536,
                          "resume_every": 256, "seed": 0, "algo": "lagrangian_ppo"},
                      fairness={"alpha": 1.0, "lambda_lr": 0.5})
    disarm = interrupt_after(900)
    with pytest.raises(Interrupted):
        train_lagrangian_ppo(cfg, save_dir=tmp_path)
    state = json.loads((P.resume_point(tmp_path) / "state.json").read_text())
    assert state["lambda_history"] and state["lambda"] == state["lambda_history"][-1][2]

    disarm()
    _, _, history = train_lagrangian_ppo(cfg, save_dir=tmp_path)
    n = len(state["lambda_history"])
    assert [tuple(h) for h in state["lambda_history"]] == history[:n]    # continued, not restarted
    assert len(history) > n


def test_parallel_training_envs_play_different_episodes():
    cfg = load_config(env={"shifts_per_episode": 3, "warmup_shifts": 1}, rl={"seed": 0})
    venv = P.build_vec_env(cfg, 4, 0, norm_reward=False)
    venv.seed(0)
    venv.reset()
    for _ in range(4):                                    # into the second episode
        venv.step(np.zeros((4, 3), np.float32))
    seeds = [e.unwrapped._episode_seed for e in venv.venv.envs]
    assert len(set(seeds)) == 4
