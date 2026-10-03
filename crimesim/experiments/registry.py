"""Definitions of the policies evaluated in the paper and helpers to train /
load / roll them out with on-disk caching so ``run_all`` is resumable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from crimesim.config import Config, load_config
from crimesim.eval.rollout import rollout
from crimesim.rl.baselines import make_baselines
from crimesim.rl.constrained_ppo import train_lagrangian_ppo, train_penalized_ppo
from crimesim.rl.ppo import load_obs_norm, train_ppo

RUNS = Path("runs")


def rl_specs(base_cfg_path: str, seeds: list[int], penalties: list[float]) -> dict:
    """policy_name -> (config, algo) for every RL policy in the study."""
    specs = {}
    for s in seeds:
        specs[f"ppo_s{s}"] = (
            load_config(base_cfg_path, rl={"seed": s, "algo": "ppo"}, env={"seed": s}), "ppo")
        specs[f"ppo_oracle_s{s}"] = (
            load_config(base_cfg_path, rl={"seed": s, "algo": "ppo"},
                        env={"seed": s, "reward_on": "latent_coverage",
                             "observe_latent": True}), "ppo")
        for p in penalties:
            specs[f"pen_p{p}_s{s}"] = (
                load_config(base_cfg_path, rl={"seed": s, "algo": "penalized_ppo"},
                            env={"seed": s}, fairness={"penalty": p}), "penalized_ppo")
    return specs


def ensure_trained(name: str, cfg: Config, algo: str, *, force: bool = False) -> Path:
    out = RUNS / name
    if (out / "ppo_model.zip").exists() and not force:
        saved = out / "config.yaml"
        if saved.exists() and load_config(saved).to_dict() != cfg.to_dict():
            # PIVOT_AUDIT.md F2: never silently reuse a model trained with another
            # config (e.g. a different timestep budget) under the same name
            raise RuntimeError(f"{out} was trained with a different config; "
                               "use a new run name or force=True")
        return out
    if algo == "penalized_ppo":
        _, vn, _ = train_penalized_ppo(cfg, save_dir=out, verbose=0)
    elif algo == "lagrangian_ppo":
        _, vn, _ = train_lagrangian_ppo(cfg, save_dir=out, verbose=0)
    else:
        _, vn = train_ppo(cfg, save_dir=out, verbose=0)
    vn.close()
    return out


def eval_policy(name: str, out: Path, eval_cfg: Config, seeds: list[int],
                tag: str) -> pd.DataFrame:
    cache = out / f"eval_{tag}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    model = PPO.load(out / "ppo_model")
    obs_norm = load_obs_norm(out / "obs_rms.npz")
    df = rollout(model, eval_cfg, seeds=seeds, keep_arrays=True,
                 obs_norm=obs_norm).assign(policy=name)
    df.to_pickle(cache)
    return df


def eval_baselines(eval_cfg: Config, seeds: list[int], cache_dir: Path,
                   tag: str) -> pd.DataFrame:
    cache = cache_dir / f"baselines_{tag}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frames = [rollout(p, eval_cfg, seeds=seeds, keep_arrays=True)
              for p in make_baselines().values()]
    df = pd.concat(frames, ignore_index=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def pool_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``ppo_s0/ppo_s1/...`` -> ``ppo`` and ``lag_a1.0_s*`` -> ``lag_a1.0``."""
    fam = df["policy"].str.replace(r"_s\d+$", "", regex=True)
    return df.assign(policy=fam)
