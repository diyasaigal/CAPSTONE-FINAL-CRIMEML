"""PPO training / loading for the PolicingEnv (Stable-Baselines3).

``train_ppo`` returns ``(model, vecnormalize)``. The VecNormalize statistics must
travel with the model at evaluation time (pass ``vecnorm`` to
``crimesim.eval.rollout.rollout``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from crimesim.config import Config
from crimesim.env.policing_env import PolicingEnv


def make_env(cfg: Config, seed: int):
    def _thunk():
        env = PolicingEnv(cfg)
        env.reset(seed=seed)
        return Monitor(env)
    return _thunk


def build_vec_env(cfg: Config, n_envs: int, seed: int, *, norm_reward: bool):
    thunks = [make_env(cfg, seed + i) for i in range(n_envs)]
    # DummyVecEnv (in-process, serial) beats SubprocVecEnv here: the env step is
    # cheap NumPy and the ~200-float observation makes per-step IPC the bottleneck.
    venv = DummyVecEnv(thunks)
    return VecNormalize(venv, norm_obs=True, norm_reward=norm_reward,
                        clip_obs=10.0, gamma=cfg.rl.gamma)


def train_ppo(cfg: Config, *, save_dir: str | Path | None = None, verbose: int = 0):
    rl = cfg.rl
    # coverage rewards are already in [0,1]; don't normalise them.
    norm_reward = cfg.env.reward_on not in ("captured", "recorded_coverage", "latent_coverage")
    venv = build_vec_env(cfg, rl.n_envs, rl.seed, norm_reward=norm_reward)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=rl.n_steps // rl.n_envs if rl.n_steps >= rl.n_envs else rl.n_steps,
        batch_size=rl.batch_size, gamma=rl.gamma, gae_lambda=rl.gae_lambda,
        learning_rate=rl.learning_rate, ent_coef=rl.ent_coef,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=rl.seed, verbose=verbose,
    )
    cb = None
    if rl.checkpoint_every > 0 and save_dir is not None:
        cb = SnapshotCallback(rl.checkpoint_every, Path(save_dir) / "checkpoints")
    model.learn(total_timesteps=rl.total_timesteps, callback=cb, progress_bar=False)

    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir / "ppo_model")
        save_obs_rms(venv, save_dir / "obs_rms.npz")
        cfg.save(save_dir / "config.yaml")
    return model, venv


class SnapshotCallback(BaseCallback):
    """Save model + observation statistics to ``<dir>/<steps>/`` every ``every`` steps."""

    def __init__(self, every: int, out_dir: Path):
        super().__init__()
        self.every, self.out_dir, self._next = every, Path(out_dir), every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            d = self.out_dir / str(self._next)
            d.mkdir(parents=True, exist_ok=True)
            self.model.save(d / "ppo_model")
            save_obs_rms(self.model.get_env(), d / "obs_rms.npz")
            self._next += self.every
        return True


def save_obs_rms(venv: VecNormalize, path) -> None:
    np.savez(path, mean=venv.obs_rms.mean, var=venv.obs_rms.var,
             epsilon=np.array(venv.epsilon), clip=np.array(venv.clip_obs))


def load_obs_norm(path):
    """Return a callable ``obs -> normalised obs`` matching VecNormalize."""
    z = np.load(path)
    mean, var, eps, clip = z["mean"], z["var"], float(z["epsilon"]), float(z["clip"])
    def norm(obs):
        return np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip)
    return norm


def load_ppo(save_dir: str | Path):
    save_dir = Path(save_dir)
    model = PPO.load(save_dir / "ppo_model")
    return model
