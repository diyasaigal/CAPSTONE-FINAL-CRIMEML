import crimesim._env_setup  # noqa: F401
"""Figures, tables and inline numbers for the journal additions.

Reads results/theory/ (scripts/analytic_model.py), results/final/reference_*.csv
(scripts/reference_robustness.py), results/journal/ (scripts/summarise_journal.py)
and results/e5_400k_e300n20_sens.csv (scripts/journal_runs.py). Writes
  figures/fig_theory.pdf, fig_optimisation.pdf, fig_sensitivity.pdf
  paper/tables/reference.tex, optimisation.tex, sensitivity.tex
  paper/numbers_journal.tex
Sections whose inputs are missing are skipped with a message.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).parent))
import run_ablations  # noqa: E402
from make_paper_assets import C, FINAL, GRID, INK, INK2, NAME, TAB, ci, save  # noqa: E402

from crimesim import theory as th  # noqa: E402
from crimesim.config import load_config  # noqa: E402

THEORY, JOURNAL = Path("results/theory"), Path("results/journal")
SENS_CSV = Path("results/e5_400k_e300n20_sens.csv")
N: dict[str, str] = {}


# ------------------------------------------------------------------ mean-field model
def theory():
    cmp = pd.read_csv(THEORY / "compare.csv")
    info = json.loads((THEORY / "summary.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"wspace": 0.28})
    for v, col, lab in [("base", C["rl"], "patrol raises detection"),
                        ("const_detect", C["parity"], "constant detection")]:
        g = cmp[cmp.variant == v]
        a1.plot(g.beta, g.model_x_oracle, "-", color=col, label=lab)
        a1.errorbar(g.beta, g.sim_x_oracle, yerr=[g.sim_x_oracle - g.sim_x_oracle_lo,
                                                  g.sim_x_oracle_hi - g.sim_x_oracle],
                    fmt="o", ms=4.5, color=col, mec="white", mew=0.7, lw=1.0)
    a1.axhline(1, color=INK, lw=0.9, ls=":")
    a1.set_yscale("log")
    a1.set_yticks([0.5, 1, 2, 4, 8]); a1.set_yticklabels(["0.5", "1", "2", "4", "8"])
    a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_xlabel(r"Concentration exponent $\beta$ of the rule")
    a1.set_ylabel("Disparity × oracle (log scale)")
    a1.set_title("(a) Disparity of a fixed rule")
    a1.legend(fontsize=6.2, loc="upper left", title="line = model, dots = simulator", title_fontsize=6.2)

    g = cmp[cmp.variant == "base"]
    for mcol, scol, col, ls, lab in [("model_rec_cov", "sim_rec_cov", C["rl"], "-", "recorded coverage (the reward)"),
                                     ("model_lat_cov", "sim_lat_cov", INK, "--", "true-crime coverage")]:
        a2.plot(g.beta, g[mcol] * 100, ls, color=col, label=lab)
        a2.plot(g.beta, g[scol] * 100, "o", ms=4.5, color=col, mec="white", mew=0.7)
    b_lat = g.loc[g.sim_lat_cov.idxmax(), "beta"]
    a2.axvline(b_lat, color=GRID, lw=1.0)
    a2.annotate("true coverage\npeaks here", (b_lat, g.sim_lat_cov.max() * 100), xytext=(6, -22),
                textcoords="offset points", fontsize=6.3, color=INK2)
    a2.set_xlabel(r"Concentration exponent $\beta$ of the rule")
    a2.set_ylabel("Coverage (%)")
    a2.set_title("(b) Coverage in the base setting")
    a2.legend(fontsize=6.2, loc="center right")
    save(fig, "fig_theory")

    base, const = cmp[cmp.variant == "base"].set_index("beta"), cmp[cmp.variant == "const_detect"].set_index("beta")
    eb, ec = info["base"]["elasticity_recorded_u1"], info["const_detect"]["elasticity_recorded_u1"]
    err = np.abs(np.log(cmp.model_disparity) - np.log(cmp.sim_disparity))
    N.update({
        "ThEpsBase": f"{eb:+.2f}", "ThEpsConst": f"{ec:+.2f}",
        "ThEpsDetect": f"{eb - ec:.2f}", "ThEpsDeter": f"{-ec:.2f}",
        "ThBetaCrit": f"{1 / eb:.1f}",
        "ThAmpOne": f"{1 / (1 - eb):.2f}", "ThAmpConstOne": f"{1 / (1 - ec):.2f}",
        "ThRtwo": f"{info['fit']['r2_log_x_oracle']:.2f}",
        "ThDispErr": f"{(np.exp(err.mean()) - 1) * 100:.0f}", "ThDispErrMax": f"{(np.exp(err.max()) - 1) * 100:.0f}",
        "ThNrules": str(len(cmp)),
        "ThLamRatio": f"{info['lambda_top_over_bottom']:.1f}",
        "ThBetaRecBase": f"{info['base']['beta_max_rec_cov_sim']:g}",
        "ThBetaLatBase": f"{info['base']['beta_max_lat_cov_sim']:g}",
        "ThBetaRecConst": f"{info['const_detect']['beta_max_rec_cov_sim']:g}",
        "ThSimBaseFour": f"{base.loc[4.0, 'sim_x_oracle']:.1f}", "ThModBaseFour": f"{base.loc[4.0, 'model_x_oracle']:.1f}",
        "ThSimConstFour": f"{const.loc[4.0, 'sim_x_oracle']:.1f}", "ThModConstFour": f"{const.loc[4.0, 'model_x_oracle']:.1f}",
        "ThSimBaseOne": f"{base.loc[1.0, 'sim_x_oracle']:.2f}", "ThSimConstOne": f"{const.loc[1.0, 'sim_x_oracle']:.2f}",
        "ThRecGainBase": f"{(base.sim_rec_cov.max() / base.loc[1.0, 'sim_rec_cov'] - 1) * 100:.0f}",
        "ThRecGainConst": f"{(const.sim_rec_cov.max() / const.loc[1.0, 'sim_rec_cov'] - 1) * 100:.0f}",
        "ThConstBetaOneFiveX": f"{const.loc[1.5, 'sim_x_oracle']:.1f}",
        "ThConstFlat": f"{(1 - const.loc[1.5:, 'sim_lat_cov'].min() / const.sim_lat_cov.max()) * 100:.1f}",
    })


# ------------------------------------------------------------------ reference robustness
def reference():
    rr = pd.read_csv(FINAL / "reference_robustness.csv")
    free = pd.read_csv(FINAL / "reference_free.csv").set_index("family")
    refs = ["GreedyOracle", "Uniform", "EWMA-like rule", "best fixed rule (latent)"]
    wide = rr.pivot(index="family", columns="reference", values="x_ref")
    fams = ["ppo", "best_rule_recorded", "ppo_oracle", "ewma_hotspot", "oracle", "best_rule_latent", "uniform",
            "pen_p0.06", "penref_w0.15"]
    rows = []
    for f in fams:
        r = free.loc[f]
        rows.append(f"{NAME[f].replace(' (reference)', '')} & "
                    + " & ".join(f"{wide.loc[f, c]:.2f}" for c in refs)
                    + f" & {r.top_excess:.2f} & {r.need_tv:.2f} \\\\")
    (TAB / "reference.tex").write_text("\n".join(rows) + "\n")
    g = lambda fam, ref: wide.loc[fam, ref]
    N.update({
        "RefUniPPO": f"{g('ppo', 'Uniform'):.1f}", "RefEwmaPPO": f"{g('ppo', 'EWMA-like rule'):.1f}",
        "RefBrlPPO": f"{g('ppo', 'best fixed rule (latent)'):.1f}",
        "TopPPO": f"{free.loc['ppo', 'top_excess']:.2f}", "TopPPOLo": f"{free.loc['ppo', 'top_excess_lo']:.2f}",
        "TopPPOHi": f"{free.loc['ppo', 'top_excess_hi']:.2f}",
        "TopOR": f"{free.loc['oracle', 'top_excess']:.2f}", "TopEWMA": f"{free.loc['ewma_hotspot', 'top_excess']:.2f}",
        "TopBRR": f"{free.loc['best_rule_recorded', 'top_excess']:.2f}",
        "TopBRL": f"{free.loc['best_rule_latent', 'top_excess']:.2f}",
        "TopUNI": f"{free.loc['uniform', 'top_excess']:.2f}",
        "TopTarget": f"{free.loc['penref_w0.15', 'top_excess']:.2f}",
        "TvPPO": f"{free.loc['ppo', 'need_tv']:.2f}", "TvOR": f"{free.loc['oracle', 'need_tv']:.2f}",
        "TvUNI": f"{free.loc['uniform', 'need_tv']:.2f}",
    })


# ------------------------------------------------------------------ optimisation check
HP_NAME = {"ppo": "PPO, default (400k)", "ppo_lr1e-3": "learning rate $10^{-3}$",
           "ppo_lr1e-4": "learning rate $10^{-4}$", "ppo_ent0": "no entropy bonus",
           "ppo_bigbatch": "8192 steps per update", "ppo_long": "default, 1.2M steps"}


def optimisation():
    opt = pd.read_csv(JOURNAL / "optimisation.csv").set_index("family")
    per = pd.read_csv(JOURNAL / "per_model.csv")
    lc = pd.read_csv(JOURNAL / "learning_curve.csv")
    info = json.loads((JOURNAL / "reward_disparity.json").read_text())
    summ = pd.read_csv(FINAL / "e1_e4_summary.csv").set_index("family")
    brr = summ.loc["best_rule_recorded"]

    rows = [f"{HP_NAME[f]} & {int(opt.loc[f, 'n_train_seeds'])} & {ci(opt.loc[f])} & "
            f"{opt.loc[f, 'rec_cov']:.4f} & {opt.loc[f, 'lat_cov']:.4f} \\\\"
            for f in HP_NAME if f in opt.index]
    rows.append("\\midrule")
    rows.append(f"Best fixed rule, recorded cov. & -- & {ci(brr)} & {brr.rec_cov:.4f} & {brr.lat_cov:.4f} \\\\")
    (TAB / "optimisation.tex").write_text("\n".join(rows) + "\n")

    cur = lc[lc.family == "ppo_long"]
    mean = cur.groupby("steps")[["rec_cov", "dlog"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw={"wspace": 0.42})
    for ax, col, tr, lab in [(axes[0], "rec_cov", lambda v: v * 100, "Recorded coverage (%)"),
                             (axes[1], "dlog", np.exp, "Disparity × oracle")]:
        for s, g in cur.groupby("tseed"):
            ax.plot(g.steps / 1e6, tr(g[col]), color=C["rl"], lw=0.7, alpha=0.45)
        ax.plot(mean.index / 1e6, tr(mean[col]), color=C["rl"], lw=1.8, label="PPO, mean of 3 seeds")
        ax.axvline(0.4, color=GRID, lw=1.0)
        ax.set_xlabel("Training steps (millions)")
        ax.set_ylabel(lab)
    axes[0].axhline(brr.rec_cov * 100, color=INK2, lw=0.9, ls="--")
    axes[0].annotate("best fixed rule", (0.65, brr.rec_cov * 100), xytext=(0, -9), textcoords="offset points",
                     fontsize=6.2, color=INK2)
    axes[0].set_title("(a) Reward")
    axes[1].axhline(1, color=INK, lw=0.9, ls=":")
    axes[1].set_title("(b) Disparity")
    ax = axes[2]
    mk = {"ppo": "o", "ppo_long": "s", "ppo_lr1e-3": "^", "ppo_lr1e-4": "v", "ppo_ent0": "D", "ppo_bigbatch": "P"}
    for f, g in per.groupby("family", sort=False):
        ax.plot(g.rec_cov * 100, g.x_ref, mk.get(f, "o"), ms=4.5, color=C["rl"], mec="white", mew=0.6,
                label=HP_NAME[f].replace("PPO, ", ""))
    ax.plot(brr.rec_cov * 100, brr.x_ref, "*", ms=9, color=INK2, mec="white", mew=0.6, label="best fixed rule")
    ax.set_yscale("log")
    ax.set_yticks([2, 4, 8, 16]); ax.set_yticklabels(["2", "4", "8", "16"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Recorded coverage (%)")
    ax.set_ylabel("Disparity × oracle (log)")
    ax.set_title("(c) Every trained model")
    ax.legend(fontsize=5.4, loc="upper left", handletextpad=0.2, borderaxespad=0.1)
    save(fig, "fig_optimisation")

    at = lambda k: mean.loc[mean.index[np.abs(mean.index - k).argmin()]]
    hp = opt.loc[[f for f in HP_NAME if f in opt.index]]
    N.update({
        "LongX": f"{opt.loc['ppo_long', 'x_ref']:.1f}", "LongXLo": f"{opt.loc['ppo_long', 'x_ref_lo']:.1f}",
        "LongXHi": f"{opt.loc['ppo_long', 'x_ref_hi']:.1f}",
        "LongRec": f"{opt.loc['ppo_long', 'rec_cov']:.4f}", "LongLat": f"{opt.loc['ppo_long', 'lat_cov']:.4f}",
        "ThreeX": f"{opt.loc['ppo', 'x_ref']:.1f}", "ThreeRec": f"{opt.loc['ppo', 'rec_cov']:.4f}",
        "CurveRecStart": f"{at(100_000).rec_cov:.4f}", "CurveRecEnd": f"{at(1_200_000).rec_cov:.4f}",
        "CurveXStart": f"{np.exp(at(100_000).dlog):.1f}", "CurveXMid": f"{np.exp(at(400_000).dlog):.1f}",
        "CurveXEnd": f"{np.exp(at(1_200_000).dlog):.1f}",
        "HpXMin": f"{hp.x_ref.min():.1f}", "HpXMax": f"{hp.x_ref.max():.1f}",
        "HpLoMin": f"{hp.x_ref_lo.min():.1f}",
        "HpRecMax": f"{hp.rec_cov.max():.4f}", "HpLatMax": f"{hp.lat_cov.max():.4f}",
        "HpN": str(info["n_models"]),
        "CurveRecMid": f"{at(400_000).rec_cov:.4f}",
        "HpRecMin": f"{hp.rec_cov.min():.4f}", "HpLatMin": f"{hp.lat_cov.min():.4f}",
        "BestModelRec": f"{per.rec_cov.max():.4f}", "BestModelX": f"{per.loc[per.rec_cov.idxmax(), 'x_ref']:.1f}",
        "HpRho": f"{info['spearman_reward_vs_log_disparity']:.2f}",
        "HpRhoP": ("<0.001" if info["p"] < 0.001 else f"={info['p']:.3f}"),
        "SnapRho": f"{info['spearman_snapshots']:.2f}", "SnapN": str(info["n_snapshots"]),
    })
    lag = [f for f in opt.index if f.startswith("lag_")]
    if lag:
        r = opt.loc[lag[0]]
        N.update({"LagX": f"{r.x_ref:.2f}", "LagXLo": f"{r.x_ref_lo:.2f}", "LagXHi": f"{r.x_ref_hi:.2f}",
                  "LagLat": f"{r.lat_cov:.4f}"})


# ------------------------------------------------------------------ sensitivity
SENS_NAME = {"base": "Base", "low_detect_gain": "Detection gain 0.25", "high_detect_gain": "Detection gain 0.80",
             "low_p_report": "Baseline reporting 0.15", "high_p_report": "Baseline reporting 0.50",
             "weak_deterrence": "Deterrence slope 0.10", "strong_deterrence": "Deterrence slope 0.30",
             "no_socio_obs": "No socio-economic inputs"}


# label offsets (points) where the default placement would collide
LABEL_AT = {"low_detect_gain": (4, 11), "low_p_report": (-5, 4), "no_socio_obs": (4, -7), "strong_deterrence": (-5, 4)}


def sensitivity():
    e5 = pd.read_csv(SENS_CSV)
    ppo = e5[e5.kind == "ppo"].set_index("variant")
    orc = e5[e5.kind == "oracle"].set_index("variant")
    rows, eps = [], {}
    for v, lab in SENS_NAME.items():
        det = load_config("configs/base.yaml", **run_ablations._merge(run_ablations.VARIANTS[v], {})).detection
        eps[v] = th.elasticity(1.0, det)
        r = ppo.loc[v]
        rows.append(f"{lab} & ${eps[v]:+.2f}$ & {orc.loc[v, 'disparity']:.2f} & {ci(r)} & "
                    f"{r.proxy_gap * 100:.2f} & {r.rec_distortion:.2f} \\\\")
        if v == "base":
            rows.append("\\midrule")
    (TAB / "sensitivity.tex").write_text("\n".join(rows) + "\n")

    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    for v, lab in SENS_NAME.items():
        r = ppo.loc[v]
        ax.plot([eps[v], eps[v]], [r.x_ref_lo, r.x_ref_hi], color=C["rl"], lw=1.3)
        ax.plot(eps[v], r.x_ref, "o", ms=5.5, color=INK if v == "base" else C["rl"], mec="white", mew=0.8, zorder=3)
        ax.annotate(lab.replace("Baseline reporting", "report").replace("Detection gain", "gain")
                    .replace("Deterrence slope", "deter.").replace("No socio-economic inputs", "no socio."),
                    (eps[v], r.x_ref), xytext=LABEL_AT.get(v, (4, 3)), textcoords="offset points",
                    ha="right" if LABEL_AT.get(v, (4, 3))[0] < 0 else "left", fontsize=5.8, color=INK2)
    ax.axhline(1, color=INK, lw=0.9, ls=":")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8]); ax.set_yticklabels(["1", "2", "4", "8"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"Signal elasticity $\varepsilon$ at proportional patrol (model)")
    ax.set_ylabel("PPO disparity × own oracle")
    save(fig, "fig_sensitivity")

    vs = [v for v in SENS_NAME if v != "no_socio_obs"]
    rho, p = sps.spearmanr([eps[v] for v in vs], [ppo.loc[v, "dlog"] for v in vs])
    N.update({
        "SensMin": f"{ppo.x_ref.min():.1f}", "SensMax": f"{ppo.x_ref.max():.1f}",
        "SensLoMin": f"{ppo.x_ref_lo.min():.2f}",
        "SensN": str(len(SENS_NAME) - 1),
        "SensRho": f"{rho:.2f}", "SensRhoP": f"{p:.3f}",
        "SensBase": f"{ppo.loc['base', 'x_ref']:.1f}",
        **{"Sens" + "".join(w.capitalize() for w in v.split("_")): f"{ppo.loc[v, 'x_ref']:.1f}"
           for v in SENS_NAME if v != "base"},
        **{"Eps" + "".join(w.capitalize() for w in v.split("_")): f"{eps[v]:+.2f}"
           for v in SENS_NAME if v not in ("base", "no_socio_obs")},
    })


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    for name, fn, needs in [("theory", theory, [THEORY / "compare.csv"]),
                            ("reference", reference, [FINAL / "reference_free.csv"]),
                            ("optimisation", optimisation, [JOURNAL / "optimisation.csv",
                                                            JOURNAL / "learning_curve.csv"]),
                            ("sensitivity", sensitivity, [SENS_CSV])]:
        if all(p.exists() for p in needs):
            fn()
            print(f"{name}: ok")
        else:
            print(f"{name}: SKIPPED, missing {[str(p) for p in needs if not p.exists()]}")
    body = ["% Generated by scripts/make_journal_assets.py -- do not edit by hand."]
    for k, val in N.items():
        assert k.isalpha(), k
        body.append(f"\\newcommand{{\\n{k}}}{{{val}}}")
    Path("paper/numbers_journal.tex").write_text("\n".join(body) + "\n")
    print(f"{len(N)} numbers")


if __name__ == "__main__":
    main()
