import crimesim._env_setup  # noqa: F401
"""E5 -- robustness / ablations.

For each variant we train a naive PPO and a penalised PPO (fixed penalty,
``--penalty``) on several training seeds, evaluate on held-out seeds, and score
every policy against that variant's own GreedyOracle on the same eval seeds
(``crimesim.eval.oracle_relative``). Resumable: trained models and rollouts are
cached, and run names / output files carry the timestep budget and eval-seed
set, so runs at different budgets never collide (PIVOT_AUDIT.md F2).

Observation-channel controls:
  * ``perfect_obs``  -- p_report=1, detect_gain=0: every latent crime is
    recorded, so patrol no longer inflates the record.
  * ``const_detect`` -- p_report=0.667, detect_gain=0: detection is again
    independent of patrol, but at the latent-weighted detection rate that
    Uniform patrol achieves in the base setting, so the agent sees about as
    much data as in ``base``. Separates "contamination" from "more data".
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from crimesim.abm.regions import load_region_info
from crimesim.config import load_config
from crimesim.eval.incidence import long_run_need, need_controlled_incidence
from crimesim.eval.oracle_relative import (add_reference_shift, episode_table,
                                           reference_disparity, relative_summary)
from crimesim.eval.rollout import rollout
from crimesim.experiments.registry import ensure_trained, eval_policy
from crimesim.rl.baselines import GreedyOracle, UniformPolicy

RESULTS = Path("results")

VARIANTS = {
    "base":            {},
    "perfect_obs":     {"detection": {"p_report": 1.0, "detect_gain": 0.0}},
    "const_detect":    {"detection": {"p_report": 0.667, "detect_gain": 0.0}},
    "low_detect_gain": {"detection": {"detect_gain": 0.25}},
    "high_detect_gain":{"detection": {"detect_gain": 0.8}},
    "weak_deterrence": {"detection": {"deterrence_slope": 0.10}},
    "strong_deterrence":{"detection": {"deterrence_slope": 0.30}},
    "no_socio_obs":    {"env": {"include_socioeconomic_obs": False}},
    "low_p_report":    {"detection": {"p_report": 0.15}},
    "high_p_report":   {"detection": {"p_report": 0.50}},
    "budget_10":       {"env": {"n_units": 10}},
    "budget_40":       {"env": {"n_units": 40}},
    "history_1":       {"env": {"history": 1}},
    "history_8":       {"env": {"history": 8}},
}


def reference_rollouts(vname: str, ecfg, seeds: list[int], suffix: str) -> pd.DataFrame:
    """GreedyOracle + Uniform under this variant on the eval seeds (cached)."""
    cache = RESULTS / f"e5_refs_{vname}_{suffix}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = pd.concat([rollout(p, ecfg, seeds, keep_arrays=True)
                    for p in (GreedyOracle(), UniformPolicy())], ignore_index=True)
    df.to_pickle(cache)
    return df


KINDS = {"ppo": lambda pen: {"rl": {"algo": "ppo"}},
         "fair": lambda pen: {"rl": {"algo": "penalized_ppo"}, "fairness": {"penalty": pen}}}


def train_specs(variants: list[str], seeds: list[int], timesteps: int,
                penalty: float, kinds=tuple(KINDS)) -> dict:
    """run name -> (training config, algo) for every E5 model."""
    specs = {}
    for vname in variants:
        for kind in kinds:
            extra = KINDS[kind]
            for s in seeds:
                merged = _merge(VARIANTS[vname], {"rl": {"seed": s, "total_timesteps": timesteps},
                                                  "env": {"seed": s}}, extra(penalty))
                cfg = load_config("configs/base.yaml", **merged)
                specs[f"abl_{vname}_{kind}_{timesteps // 1000}k_s{s}"] = (cfg, cfg.rl.algo)
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", "--seeds", dest="seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(300, 320)))
    ap.add_argument("--timesteps", type=int, default=400000)
    ap.add_argument("--penalty", type=float, default=0.06)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    ap.add_argument("--out-tag", default="", help="suffix for the output files")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    region = load_region_info(20)
    budget = f"{args.timesteps // 1000}k"
    suffix = f"e{args.eval_seeds[0]}n{len(args.eval_seeds)}"
    tag = f"abl_{suffix}"

    tables, raw, inc = [], [], []
    variants = {k: VARIANTS[k] for k in (args.only or VARIANTS)}
    for vname, over in variants.items():
        ecfg = load_config("configs/base.yaml", **_merge(over, {}))
        refs = reference_rollouts(vname, ecfg, args.eval_seeds, suffix)
        orc = episode_table(refs[refs.policy == "oracle"], region, variant=vname, kind="oracle", tseed=-1)
        ref = reference_disparity(orc)
        need = long_run_need(refs[refs.policy == "uniform"])
        vt = [orc]
        specs = train_specs([vname], args.seeds, args.timesteps, args.penalty, args.kinds)
        for kind in args.kinds:
            frames = []
            for s in args.seeds:
                name = f"abl_{vname}_{kind}_{budget}_s{s}"
                cfg, algo = specs[name]
                t0 = time.time()
                out = ensure_trained(name, cfg, algo)
                t1 = time.time()
                df = eval_policy(name, out, ecfg, args.eval_seeds, tag)
                print(f"{name}: train {t1 - t0:.0f}s, eval {time.time() - t1:.0f}s", flush=True)
                df = df.assign(variant=vname, kind=kind, tseed=s)
                frames.append(df)
                vt.append(episode_table(df, region, variant=vname, kind=kind, tseed=s))
            g = pd.concat(frames, ignore_index=True)
            raw.append(g)
            h = need_controlled_incidence(g, region, need)["hardship"]
            inc.append({"variant": vname, "kind": kind, "hardship_coef": h["coef"],
                        "hardship_t": h["t"], "hardship_p": h["p"]})
        eps = add_reference_shift(pd.concat(vt, ignore_index=True), ref)
        tables.append((eps, relative_summary(eps, ["variant", "kind"], ref)))

    stem = f"e5_{budget}_{suffix}{args.out_tag}"
    pd.concat(raw, ignore_index=True).to_pickle(RESULTS / f"{stem}_raw.pkl")
    pd.concat([e for e, _ in tables], ignore_index=True).to_csv(
        RESULTS / f"{stem}_episodes.csv", index=False)
    out = (pd.concat([s for _, s in tables], ignore_index=True)
           .merge(pd.DataFrame(inc), on=["variant", "kind"], how="left"))
    out.to_csv(RESULTS / f"{stem}.csv", index=False)
    cols = ["variant", "kind", "n_train_seeds", "n_eval_seeds", "x_ref", "x_ref_lo", "x_ref_hi",
            "disparity", "lat_cov", "proxy_gap", "rec_distortion", "hardship_t"]
    pd.set_option("display.width", 220)
    print(f"\n== E5 ({budget}, eval {suffix}): disparity x own-variant GreedyOracle ==")
    print(out[cols].round(3).to_string(index=False))


def _merge(*dicts):
    m: dict = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict):
                m.setdefault(k, {}).update(v)
            else:
                m[k] = v
    return m


if __name__ == "__main__":
    main()
