"""PPO training / loading for the PolicingEnv (Stable-Baselines3).

``train_ppo`` returns ``(model, vecnormalize)``. The VecNormalize statistics must
travel with the model at evaluation time (pass ``vecnorm`` to
``crimesim.eval.rollout.rollout``).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from crimesim.config import Config, load_config
from crimesim.env.policing_env import PolicingEnv


def make_env(cfg: Config, seed: int):
    def _thunk():
        env = PolicingEnv(cfg)
        env.reset(seed=seed)
        return Monitor(env)
    return _thunk


def raw_vec_env(cfg: Config, n_envs: int, seed: int):
    # DummyVecEnv (in-process, serial) beats SubprocVecEnv here: the env step is
    # cheap NumPy and the ~200-float observation makes per-step IPC the bottleneck.
    return DummyVecEnv([make_env(cfg, seed + i) for i in range(n_envs)])


def normalize(venv, cfg: Config, *, norm_reward: bool) -> VecNormalize:
    return VecNormalize(venv, norm_obs=True, norm_reward=norm_reward,
                        clip_obs=10.0, gamma=cfg.rl.gamma)


def build_vec_env(cfg: Config, n_envs: int, seed: int, *, norm_reward: bool):
    return normalize(raw_vec_env(cfg, n_envs, seed), cfg, norm_reward=norm_reward)


# ------------------------------------------------------------------ resumable training
# With ``rl.resume_every > 0`` and a ``save_dir``, training writes a resume point to
# ``<save_dir>/resume/`` every ``resume_every`` steps: the model with its optimiser
# state, the VecNormalize statistics, the step count and any trainer state (the
# Lagrange multiplier). If the process stops, calling the trainer again with the same
# config continues from the last resume point instead of from step 0. The continued
# run is statistically, not bit-for-bit, identical to an uninterrupted one: the
# environments restart new episodes (on fresh seed streams) and the partly collected
# rollout is lost. ``resumed_at.json`` beside the final model lists where it resumed.
RESUME, RESUME_TMP = "resume", "resume_tmp"


def resume_point(save_dir) -> Path | None:
    for name in (RESUME, RESUME_TMP):
        d = Path(save_dir) / name
        if (d / "state.json").exists():
            return d
    return None


class ResumeCallback(BaseCallback):
    """Write a resume point at the first rollout start past every ``every`` steps
    (after the previous update, so nothing is half-applied). Written to a temporary
    directory first and swapped in, so a crash mid-write keeps the previous point."""

    def __init__(self, every: int, save_dir, cfg: Config, extra=None):
        super().__init__()
        self.every, self.dir, self.cfg = every, Path(save_dir), cfg
        self.extra = extra or (lambda: {})
        self._next = every

    def _on_training_start(self) -> None:
        self._next = (self.num_timesteps // self.every + 1) * self.every

    def _on_rollout_start(self) -> None:
        if self.num_timesteps >= self._next:
            self.save()
            while self._next <= self.num_timesteps:
                self._next += self.every

    def _on_step(self) -> bool:
        return True

    def save(self) -> None:
        tmp, final = self.dir / RESUME_TMP, self.dir / RESUME
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        self.model.save(tmp / "model")
        self.model.get_vec_normalize_env().save(str(tmp / "vecnormalize.pkl"))
        self.cfg.save(tmp / "config.yaml")
        (tmp / "state.json").write_text(json.dumps({"num_timesteps": int(self.num_timesteps),
                                                    **self.extra()}))
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)


def learn_resumable(cfg: Config, venv, make_model, *, norm_reward: bool, save_dir=None,
                    callbacks=(), extra_state=None, restore_state=None, verbose: int = 0):
    """Build (or resume) a PPO run on ``venv`` and train it to ``cfg.rl.total_timesteps``.

    ``make_model(vnorm)`` builds a fresh model. ``extra_state()`` returns trainer state
    to store at each resume point and ``restore_state(state)`` puts it back.
    Returns ``(model, vnorm, resumed_at)``.
    """
    resumable = save_dir is not None and cfg.rl.resume_every > 0
    point = resume_point(save_dir) if resumable else None
    resumed_at: list[int] = []
    if point is not None:
        if load_config(point / "config.yaml").to_dict() != cfg.to_dict():
            raise RuntimeError(f"{point} was written with another config; delete it to start over")
        state = json.loads((point / "state.json").read_text())
        resumed_at = state.get("resumed_at", []) + [state["num_timesteps"]]
        vnorm = VecNormalize.load(str(point / "vecnormalize.pkl"), venv)
        vnorm.training = True
        model = PPO.load(point / "model", env=vnorm, verbose=verbose)
        vnorm.seed(cfg.rl.seed + 7919 * len(resumed_at))     # fresh episode streams
        if restore_state is not None:
            restore_state(state)
    else:
        vnorm = normalize(venv, cfg, norm_reward=norm_reward)
        model = make_model(vnorm)
    cbs = list(callbacks)
    if resumable:
        cbs.append(ResumeCallback(cfg.rl.resume_every, save_dir, cfg, extra=lambda: {
            "resumed_at": resumed_at, **(extra_state() if extra_state else {})}))
    remaining = cfg.rl.total_timesteps - model.num_timesteps
    if remaining > 0:
        model.learn(total_timesteps=remaining, callback=CallbackList(cbs) if cbs else None,
                    reset_num_timesteps=point is None, progress_bar=False)
    return model, vnorm, resumed_at


def finish_resume(save_dir, resumed_at: list[int]) -> None:
    """After the final model is saved: drop the resume points, record any resumptions."""
    if save_dir is None:
        return
    for name in (RESUME, RESUME_TMP):
        shutil.rmtree(Path(save_dir) / name, ignore_errors=True)
    if resumed_at:
        (Path(save_dir) / "resumed_at.json").write_text(json.dumps(resumed_at))


def train_ppo(cfg: Config, *, save_dir: str | Path | None = None, verbose: int = 0):
    rl = cfg.rl
    # coverage rewards are already in [0,1]; don't normalise them.
    norm_reward = cfg.env.reward_on not in ("captured", "recorded_coverage", "latent_coverage")

    def make_model(venv):
        return PPO(
            "MlpPolicy", venv,
            n_steps=rl.n_steps // rl.n_envs if rl.n_steps >= rl.n_envs else rl.n_steps,
            batch_size=rl.batch_size, gamma=rl.gamma, gae_lambda=rl.gae_lambda,
            learning_rate=rl.learning_rate, ent_coef=rl.ent_coef,
            policy_kwargs=dict(net_arch=[128, 128]),
            seed=rl.seed, verbose=verbose,
        )

    cbs = []
    if rl.checkpoint_every > 0 and save_dir is not None:
        cbs.append(SnapshotCallback(rl.checkpoint_every, Path(save_dir) / "checkpoints"))
    model, venv, resumed_at = learn_resumable(
        cfg, raw_vec_env(cfg, rl.n_envs, rl.seed), make_model, norm_reward=norm_reward,
        save_dir=save_dir, callbacks=cbs, verbose=verbose)

    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_dir / "ppo_model")
        save_obs_rms(venv, save_dir / "obs_rms.npz")
        cfg.save(save_dir / "config.yaml")
        finish_resume(save_dir, resumed_at)
    return model, venv


class SnapshotCallback(BaseCallback):
    """Save model + observation statistics to ``<dir>/<steps>/`` every ``every`` steps."""

    def __init__(self, every: int, out_dir: Path):
        super().__init__()
        self.every, self.out_dir, self._next = every, Path(out_dir), every

    def _on_training_start(self) -> None:
        # after a resume, continue from the next multiple of ``every``
        self._next = (self.num_timesteps // self.every + 1) * self.every

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
