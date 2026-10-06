import crimesim._env_setup  # noqa: F401
"""Pivot Phase 6: every figure, table and inline number of the rewritten paper.

Reads ``results/final/`` (from ``scripts/summarise_400k.py``) and the E5 summary
(``results/e5_400k_e300n20.csv``); the only new rollouts are a short replay of
each trained model that records the rule parameters it chooses.

Writes, without touching the pre-pivot figures:
  figures/fig_arch.pdf, fig_e1_disparity.pdf, fig_e2_contamination.pdf,
  fig_e3_need_residual.pdf, fig_e4_frontier.pdf, fig_e5_interventions.pdf (+ .png previews)
  paper/tables/*.tex      LaTeX table bodies, \\input by paper/main.tex
  paper/numbers.tex       \\newcommand macros for every number quoted in the text
  results/final/learned_rules.csv
so the paper cannot drift from the CSVs: regenerate, never hand-edit.
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from crimesim.abm.regions import load_region_info
from crimesim.config import load_config
from crimesim.eval.incidence import long_run_need
from crimesim.eval.rollout import rollout
from crimesim.rl.ppo import load_obs_norm

sys.path.insert(0, str(Path(__file__).parent))
from summarise_400k import RL, RUNS, load_frames  # noqa: E402

FINAL = Path("results/final")
FIG = Path("figures")
TAB = Path("paper/tables")
region = load_region_info(20)

# ------------------------------------------------------------------ style
# roles, not series: reference / rule-based / unconstrained RL / parity penalty / oracle-targeted penalty
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
C = {"ref": INK, "rule": "#8a8985", "rl": "#eb6834", "parity": "#2a78d6", "target": "#1baf7a"}
plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 8.5, "axes.titleweight": "bold", "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 200, "lines.linewidth": 1.6})

NAME = {
    "oracle": "Greedy oracle (reference)", "uniform": "Uniform", "random": "Random",
    "ewma_hotspot": "PredPol (EWMA)", "rf_hotspot": "Random forest",
    "best_rule_recorded": "Best fixed rule, recorded cov.", "best_rule_latent": "Best fixed rule, true cov.",
    "ppo": "PPO (naive)", "ppo_obslat": "PPO, sees true crime", "ppo_rewlat": "PPO, rewarded on true crime",
    "ppo_oracle": "PPO, sees + rewarded on true crime",
    **{f"pen_p{p}": f"Parity penalty {p:g}" for p in [0.02, 0.06, 0.15, 0.4]},
    **{f"penref_w{w}": f"Oracle-targeted penalty {w:g}" for w in [0.05, 0.15, 0.5]}}
ROLE = {"oracle": "ref", "uniform": "rule", "random": "rule", "ewma_hotspot": "rule", "rf_hotspot": "rule",
        "best_rule_recorded": "rule", "best_rule_latent": "rule", "ppo": "rl", "ppo_obslat": "rl",
        "ppo_rewlat": "rl", "ppo_oracle": "rl",
        **{f"pen_p{p}": "parity" for p in [0.02, 0.06, 0.15, 0.4]},
        **{f"penref_w{w}": "target" for w in [0.05, 0.15, 0.5]}}
SHORT = {"oracle": "Greedy oracle", "ewma_hotspot": "PredPol", "ppo_oracle": "PPO, true crime",
         "ppo": "PPO (naive)", "best_rule_recorded": "Best rule, rec.", "pen_p0.02": "Parity 0.02",
         "penref_w0.15": "Targeted 0.15", "uniform": "Uniform"}
ORDER = ["oracle", "uniform", "random", "ewma_hotspot", "rf_hotspot", "best_rule_latent", "best_rule_recorded",
         "ppo", "ppo_obslat", "ppo_rewlat", "ppo_oracle",
         "pen_p0.02", "pen_p0.06", "pen_p0.15", "pen_p0.4", "penref_w0.05", "penref_w0.15", "penref_w0.5"]


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# ------------------------------------------------------------------ learned rule parameters
class _Recorder:
    def __init__(self, model):
        self.model, self.actions = model, []

    def predict(self, obs, deterministic=True):
        a, _ = self.model.predict(obs, deterministic=deterministic)
        self.actions.append(np.clip(a, -1, 1))
        return a, None


def learned_rules(seeds=range(200, 205)) -> pd.DataFrame:
    cache = FINAL / "learned_rules.csv"
    if cache.exists():
        return pd.read_csv(cache)
    from stable_baselines3 import PPO
    rows = []
    for fam, tseeds in RL.items():
        for s in tseeds:
            out = RUNS / f"{fam}_s{s}"
            cfg = load_config(out / "config.yaml")
            ecfg = load_config("configs/base.yaml", env={"observe_latent": cfg.env.observe_latent,
                                                         "reward_on": cfg.env.reward_on})
            rec = _Recorder(PPO.load(out / "ppo_model"))
            rollout(rec, ecfg, list(seeds), obs_norm=load_obs_norm(out / "obs_rms.npz"))
            a = np.array(rec.actions)
            rows.append({"family": fam, "tseed": s, "beta": (2 * (a[:, 0] + 1)).mean(),
                         "w": (2 * a[:, 1]).mean(), "mu": (0.5 * (a[:, 2] + 1)).mean(),
                         "w_pos_share": (a[:, 1] > 0).mean()})
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


# ------------------------------------------------------------------ figures
def fig_arch():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 120); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc="#f0efec", ec=INK2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.8,
                fontweight="bold" if bold else "normal", color=INK, linespacing=1.35)

    def arrow(p, q, color=INK2, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color,
                                     linestyle=ls, connectionstyle=f"arc3,rad={rad}"))

    def label(x, y, text, color=INK2, ha="center"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=6.3, color=color)

    box(1, 16, 19, 11, "Unseeded ABM\nlatent crime $L_r$", bold=True)
    box(29, 16, 27, 11, "Detection probability\n" r"$q_r = 0.30 + 0.55\,\min(1, u_r/1.5)$")
    box(65, 16, 21, 11, "Recorded crime\n" r"$D_r \sim \mathrm{Bin}(L_r, q_r)$")
    box(96, 16, 23, 11, "Policy $\\pi$ (PPO)\n" r"sets rule $(\beta, w, \mu)$" "\nfrom EWMA of $D$",
        fc="#fbe3d8", ec=C["rl"], bold=True)
    box(37, 34, 42, 8, "Patrol intensity $u_r$ in each of 22 districts", fc="#ffffff")
    box(37, 1, 42, 8, "Evaluation only: compare with the\n" r"Greedy oracle, $a_r \propto L_r$", fc="#ffffff",
        ec=C["ref"])

    arrow((20.5, 21.5), (28.5, 21.5))
    arrow((56.5, 21.5), (64.5, 21.5))
    arrow((86.5, 23.5), (95.5, 23.5)); label(91, 25.6, "observes")
    arrow((86.5, 19.5), (95.5, 19.5), color=C["rl"], ls="--"); label(91, 16.3, "reward\ncov$(a, D)$", C["rl"])
    arrow((107.5, 27.5), (79.5, 38), rad=0.2); label(101, 37, "allocates")
    arrow((46, 33.5), (42.5, 27.5), color=C["rl"]); label(51.5, 30.5, "raises $q_r$", C["rl"])
    arrow((36.5, 38), (10.5, 27.5), rad=0.2); label(15, 36.5, "deters (bounded)")
    arrow((10.5, 15.5), (36.5, 5), color=C["ref"], ls=":", rad=0.2); label(14, 7, "never shown\nto the policy", C["ref"])
    arrow((79.5, 5), (107.5, 15.5), color=C["ref"], ls=":", rad=0.2)
    save(fig, "fig_arch")


def fig_e1(summ, per):
    fams = [f for f in ORDER if f in set(summ.family)]
    s = summ.set_index("family").loc[fams]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.9), sharey=True,
                                 gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.06})
    y = np.arange(len(fams))[::-1]
    for yi, f in zip(y, fams):
        c = C[ROLE[f]]
        r = s.loc[f]
        a1.plot([r.x_ref_lo, r.x_ref_hi], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
        a1.plot(r.x_ref, yi, "o", ms=6.5, color=c, mec="white", mew=1.0, zorder=3)
        pts = per[per.family == f].x_ref
        a1.plot(pts, np.full(len(pts), yi - 0.3), ".", ms=3, color=c, alpha=0.9)
        a2.plot(r.lat_cov * 100, yi, "o", ms=6.5, color=c, mec="white", mew=1.0, zorder=3)
    a1.axvline(1, color=INK, lw=0.9, ls=":")
    a1.set_xscale("log")
    a1.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 8, 16])
    a1.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4", "8", "16"])
    a1.set_yticks(y); a1.set_yticklabels([NAME[f] for f in fams])
    a1.set_xlabel("Disparity relative to the greedy oracle (log scale)\n"
                  "dot = mean, bar = 95% CI, small dots = single training seeds")
    a1.set_title("(a) Hardship disparity, × oracle")
    a2.axvline(s.loc["oracle", "lat_cov"] * 100, color=INK, lw=0.9, ls=":")
    a2.set_xlabel("True-crime coverage (%)")
    a2.set_title("(b) True-crime coverage")
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    for yb in [y[fams.index("ppo")] + 0.5, y[fams.index("pen_p0.02")] + 0.5]:
        a1.axhline(yb, color=GRID, lw=0.8); a2.axhline(yb, color=GRID, lw=0.8)
    save(fig, "fig_e1_disparity")


def fig_e2(frames, summ):
    show = ["oracle", "ewma_hotspot", "ppo", "pen_p0.02", "penref_w0.15"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    for f in show:
        df = pd.concat([v for (fam, _), v in frames.items() if fam == f])
        g = (df.recorded_coverage - df.latent_coverage).groupby(df["shift"]).mean().rolling(12, center=True,
                                                                                             min_periods=1).mean()
        a1.plot(g.index, g * 100, color=C[ROLE[f]], ls="--" if f == "pen_p0.02" else "-",
                lw=1.8 if f == "ppo" else 1.3)
        a1.lines[-1].set_label(NAME[f].replace(" (reference)", ""))
    a1.legend(fontsize=6.3, loc="center right")
    a1.set_xlim(0, 139)
    a1.set_xlabel("Shift within the 140-shift episode")
    a1.set_ylabel("Recorded − true coverage (pp)")
    a1.set_title("(a) Proxy gap over the episode")

    fams = [f for f in ["uniform", "oracle", "ewma_hotspot", "ppo_oracle", "ppo", "best_rule_recorded",
                        "pen_p0.02", "penref_w0.15"]]
    s = summ.set_index("family").loc[fams]
    x = np.arange(len(fams))
    a2.bar(x, s.rec_distortion, width=0.62, color=[C[ROLE[f]] for f in fams], edgecolor="white", lw=1.0)
    a2.axhline(1, color=INK, lw=0.9, ls=":")
    a2.set_xticks(x); a2.set_xticklabels([SHORT[f] for f in fams], rotation=45, ha="right", fontsize=6.4)
    for xi, v in zip(x, s.rec_distortion):
        a2.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=6.2, color=INK2)
    a2.set_ylabel("Q5 / Q1 over-recording")
    a2.set_title("(b) Records tilted toward\nhigh-hardship districts")
    a2.grid(axis="x", visible=False)
    save(fig, "fig_e2_contamination")


def need_residual_by_quintile(frames, fam, need, groups, burn_in=0.3):
    dfs = [v for (f, _), v in frames.items() if f == fam]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["shift"] >= burn_in * df["shift"].max()]
    dep = np.stack(df.deployment.to_numpy()).astype(float)
    share = (dep / dep.sum(1, keepdims=True)).mean(0)
    res = share - need
    return np.array([res[groups == q].sum() for q in range(groups.max() + 1)])


def fig_e3(frames, e3t):
    need = long_run_need(frames[("uniform", -1)])
    groups = region.quantile_groups("hardship", 5)
    show = ["oracle", "ewma_hotspot", "ppo", "ppo_oracle", "pen_p0.02", "penref_w0.15"]
    style = {"oracle": ("-", "o"), "ewma_hotspot": (":", "s"), "ppo": ("-", "o"), "ppo_oracle": (":", "^"),
             "pen_p0.02": ("--", "o"), "penref_w0.15": ("-", "D")}
    fig, ax = plt.subplots(figsize=(3.45, 2.8))
    q = np.arange(1, 6)
    for f in show:
        r = need_residual_by_quintile(frames, f, need, groups) * 100
        ls, mk = style[f]
        ax.plot(q, r, ls=ls, marker=mk, ms=4.5, color=C[ROLE[f]], label=NAME[f].replace(" (reference)", ""),
                mec="white", mew=0.6)
    ax.axhline(0, color=INK, lw=0.9, ls=":")
    ax.set_xticks(q)
    ax.set_xlabel("District hardship quintile (5 = most disadvantaged)")
    ax.set_ylabel("Patrol share − need (pp)")
    ax.legend(fontsize=6.2, loc="upper left", handlelength=2.4)
    save(fig, "fig_e3_need_residual")


OFF = {"pen_p0.02": (2, -10), "pen_p0.06": (5, -6), "pen_p0.15": (5, -6), "pen_p0.4": (5, -3),
       "penref_w0.05": (-6, 6), "penref_w0.15": (-4, 6), "penref_w0.5": (-14, 4)}


def fig_e4(summ):
    s = summ.set_index("family")
    fig, ax = plt.subplots(figsize=(3.45, 2.9))
    for chain, role, ls in [(["ppo", "pen_p0.02", "pen_p0.06", "pen_p0.15", "pen_p0.4"], "parity", "--"),
                            (["ppo", "penref_w0.05", "penref_w0.15", "penref_w0.5"], "target", "-")]:
        xs, ys = s.loc[chain, "x_ref"], s.loc[chain, "lat_cov"] * 100
        ax.plot(xs, ys, ls=ls, color=C[role], lw=1.2, zorder=1)
        for f in chain[1:]:
            r = s.loc[f]
            ax.errorbar(r.x_ref, r.lat_cov * 100, xerr=[[r.x_ref - r.x_ref_lo], [r.x_ref_hi - r.x_ref]],
                        fmt="o", ms=5, color=C[role], mec="white", mew=0.8, capsize=0, lw=1.0, zorder=3)
            ax.annotate(f.split("_")[1].lstrip("pw"), (r.x_ref, r.lat_cov * 100), xytext=OFF.get(f, (4, -9)),
                        textcoords="offset points", fontsize=6.3, color=C[role])
    for f in ["ppo", "oracle", "uniform", "ewma_hotspot"]:
        r = s.loc[f]
        ax.plot(r.x_ref, r.lat_cov * 100, "o" if f != "ppo" else "s", ms=6, color=C[ROLE[f]],
                mec="white", mew=0.8, zorder=4)
        ax.annotate(NAME[f].replace(" (reference)", ""), (r.x_ref, r.lat_cov * 100), xytext=(5, 2),
                    textcoords="offset points", fontsize=6.4, color=INK)
    ax.axvline(1, color=INK, lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("Disparity × oracle (log scale)")
    ax.set_ylabel("True-crime coverage (%)")
    ax.plot([], [], ls="--", color=C["parity"], label="parity penalty $-w\\,\\delta$")
    ax.plot([], [], ls="-", color=C["target"], label="oracle-targeted penalty")
    ax.legend(fontsize=6.4, loc="lower right")
    save(fig, "fig_e4_frontier")


def fig_e5(e5):
    variants = [("base", "Base\n(patrol raises detection)"), ("perfect_obs", "Perfect observation\n(q = 1)"),
                ("const_detect", "Constant detection\n(q = 0.667)")]
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    for i, (v, lab) in enumerate(variants):
        for kind, role, dx in [("ppo", "rl", -0.12), ("fair", "parity", 0.12)]:
            r = e5[(e5.variant == v) & (e5.kind == kind)].iloc[0]
            ax.plot([i + dx, i + dx], [r.x_ref_lo, r.x_ref_hi], color=C[role], lw=1.6)
            ax.plot(i + dx, r.x_ref, "o", ms=6, color=C[role], mec="white", mew=0.8)
            if kind == "ppo":
                ax.annotate(f"{r.x_ref:.2f}×", (i + dx, r.x_ref), xytext=(-5, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=6.5, color=INK)
    ax.axhline(1, color=INK, lw=0.9, ls=":")
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 6]); ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "6"])
    ax.set_ylim(0.22, 8)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in variants], fontsize=6.6)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Disparity × that setting's oracle")
    ax.plot([], [], "o", color=C["rl"], label="PPO (naive)")
    ax.plot([], [], "o", color=C["parity"], label="Parity penalty 0.06")
    ax.legend(fontsize=6.4, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_e5_interventions")


# ------------------------------------------------------------------ tables + numbers
def ci(r, k="x_ref"):
    return f"{r[k]:.2f} [{r[k + '_lo']:.2f}, {r[k + '_hi']:.2f}]"


def tables(summ, e3t, att, e5, ct, rules):
    TAB.mkdir(parents=True, exist_ok=True)
    s = summ.set_index("family")
    groups = [["oracle", "uniform", "random", "ewma_hotspot", "rf_hotspot"],
              ["best_rule_latent", "best_rule_recorded"],
              ["ppo", "ppo_oracle"],
              ["pen_p0.02", "pen_p0.06", "pen_p0.15", "pen_p0.4"],
              ["penref_w0.05", "penref_w0.15", "penref_w0.5"]]
    lines = []
    for gi, g in enumerate(groups):
        if gi:
            lines.append("\\midrule")
        for f in g:
            r = s.loc[f]
            n = int(r.n_train_seeds) if ROLE[f] not in ("ref", "rule") else "--"
            x = "1 (ref.)" if f == "oracle" else ci(r)
            lines.append(f"{NAME[f].replace(' (reference)', '')} & {n} & {x} & {r.disparity:.2f} & "
                         f"{r.lat_cov:.4f} & {r.rec_cov:.4f} & {r.proxy_gap:.4f} & {r.rec_distortion:.2f} \\\\")
    (TAB / "e1.tex").write_text("\n".join(lines) + "\n")

    a = att.set_index("family")
    cell = lambda f: f"{a.loc[f, 'x_ref']:.2f} [{a.loc[f, 'x_ref_lo']:.2f}, {a.loc[f, 'x_ref_hi']:.2f}]"
    (TAB / "attribution.tex").write_text(
        f"Recorded crime only & {cell('ppo')} & {cell('ppo_rewlat')} \\\\\n"
        f"Recorded + true crime & {cell('ppo_obslat')} & {cell('ppo_oracle')} \\\\\n")

    e = e3t.set_index("family")
    rows = []
    for f in ["oracle", "uniform", "ewma_hotspot", "rf_hotspot", "best_rule_recorded", "ppo", "ppo_oracle",
              "ppo_obslat", "ppo_rewlat", "pen_p0.02", "pen_p0.4", "penref_w0.15"]:
        r = e.loc[f]
        if f == "uniform":
            rows.append("Uniform & \\multicolumn{4}{c}{constant allocation; not estimable} \\\\")
            continue
        seed = (f"[{r.seed_boot_lo * 1e3:.1f}, {r.seed_boot_hi * 1e3:.1f}] & {r.seeds_positive}"
                if isinstance(r.seeds_positive, str) else "-- & --")
        rows.append(f"{NAME[f].replace(' (reference)', '')} & ${r.pooled_coef * 1e3:+.1f}$ & "
                    f"{r.pooled_t:.1f} & {seed} \\\\")
    (TAB / "e3.tex").write_text("\n".join(rows) + "\n")

    rows = []
    for v, lab in [("base", "Base (patrol raises detection)"), ("perfect_obs", "Perfect observation"),
                   ("const_detect", "Constant detection, $q=0.667$")]:
        p = e5[(e5.variant == v) & (e5.kind == "ppo")].iloc[0]
        f = e5[(e5.variant == v) & (e5.kind == "fair")].iloc[0]
        o = e5[(e5.variant == v) & (e5.kind == "oracle")].iloc[0]
        rows.append(f"{lab} & {o.disparity:.2f} & {ci(p)} & {ci(f)} & {p.rec_distortion:.2f} \\\\")
    (TAB / "e5.tex").write_text("\n".join(rows) + "\n")

    lab = lambda x: NAME.get(x, {"E5 perfect_obs ppo": "E5 PPO, perfect obs.", "E5 const_detect ppo":
                                  "E5 PPO, constant detection", "E5 base ppo": "E5 PPO, base"}.get(x, x))
    rows = []
    for _, r in ct.iterrows():
        diff = f"{np.exp(r['diff']):.2f}$\\times$" if r.metric == "dlog" else f"${r['diff'] * 100:+.2f}$ pp"
        rng = (f"[{np.exp(r.lo):.2f}, {np.exp(r.hi):.2f}]" if r.metric == "dlog"
               else f"[{r.lo * 100:+.2f}, {r.hi * 100:+.2f}]")
        metric = {"dlog": "disparity", "lat_cov": "true cov.", "rec_cov": "recorded cov."}[r.metric]
        p = "$<0.005$" if r.p_holm < 0.005 else f"{r.p_holm:.3f}"
        rows.append(f"{lab(r.a)} vs {lab(r.b).replace(' (reference)', '')} & {metric} & {diff} & {rng} & {p} \\\\")
    (TAB / "contrasts.tex").write_text("\n".join(rows) + "\n")

    lr = rules.groupby("family")[["beta", "w", "mu", "w_pos_share"]].mean()
    rows = []
    for f in ["ppo", "ppo_obslat", "ppo_rewlat", "ppo_oracle", "pen_p0.02", "pen_p0.4", "penref_w0.15"]:
        r = lr.loc[f]
        rows.append(f"{NAME[f]} & {r.beta:.2f} & ${r.w:+.2f}$ & {r.mu:.2f} \\\\")
    rows.append("\\midrule")
    for f, (b, w, mu) in [("best_rule_recorded", (4, 2, 0)), ("best_rule_latent", (1.5, 0, 0))]:
        rows.append(f"{NAME[f]} & {b:.2f} & ${w:+.2f}$ & {mu:.2f} \\\\")
    rows.append(f"PredPol (EWMA), equivalent rule & 1.00 & $+0.00$ & 0.00 \\\\")
    (TAB / "rules.tex").write_text("\n".join(rows) + "\n")


def numbers(summ, per, e3t, att, e5, ct, rules, qres):
    s, a, e = summ.set_index("family"), att.set_index("family"), e3t.set_index("family")
    c = ct.assign(key=ct.a + "|" + ct.b + "|" + ct.metric).set_index("key")
    sp = per.groupby("family").x_ref.agg(["min", "max"])
    v = lambda var, kind: e5[(e5.variant == var) & (e5.kind == kind)].iloc[0]
    lr = rules.groupby("family")[["beta", "w", "mu", "w_pos_share"]].mean()
    N = {
        "PPOx": f"{s.loc['ppo', 'x_ref']:.1f}", "PPOxLo": f"{s.loc['ppo', 'x_ref_lo']:.1f}",
        "PPOxHi": f"{s.loc['ppo', 'x_ref_hi']:.1f}",
        "PPOseedMin": f"{sp.loc['ppo', 'min']:.1f}", "PPOseedMax": f"{sp.loc['ppo', 'max']:.1f}",
        "PPOdisp": f"{s.loc['ppo', 'disparity']:.1f}", "ORdisp": f"{s.loc['oracle', 'disparity']:.2f}",
        "PPOlat": f"{s.loc['ppo', 'lat_cov']:.4f}", "PPOrec": f"{s.loc['ppo', 'rec_cov']:.4f}",
        "ORlat": f"{s.loc['oracle', 'lat_cov']:.4f}", "UNIlat": f"{s.loc['uniform', 'lat_cov']:.4f}",
        "EWMAlat": f"{s.loc['ewma_hotspot', 'lat_cov']:.4f}", "EWMArec": f"{s.loc['ewma_hotspot', 'rec_cov']:.4f}",
        "EWMAx": f"{s.loc['ewma_hotspot', 'x_ref']:.2f}", "RFx": f"{s.loc['rf_hotspot', 'x_ref']:.2f}",
        "UNIx": f"{s.loc['uniform', 'x_ref']:.2f}",
        "PPOgap": f"{s.loc['ppo', 'proxy_gap']:.4f}", "EWMAgap": f"{s.loc['ewma_hotspot', 'proxy_gap']:.4f}",
        "ORgap": f"{s.loc['oracle', 'proxy_gap']:.4f}",
        "PPOrd": f"{s.loc['ppo', 'rec_distortion']:.2f}", "ORrd": f"{s.loc['oracle', 'rec_distortion']:.2f}",
        "EWMArd": f"{s.loc['ewma_hotspot', 'rec_distortion']:.2f}",
        "PPOvsEWMA": f"{np.exp(c.loc['ppo|ewma_hotspot|dlog', 'diff']):.1f}",
        "PPOORx": f"{s.loc['ppo_oracle', 'x_ref']:.2f}", "PPOORlat": f"{s.loc['ppo_oracle', 'lat_cov']:.4f}",
        "BRRx": f"{s.loc['best_rule_recorded', 'x_ref']:.1f}", "BRRrec": f"{s.loc['best_rule_recorded', 'rec_cov']:.4f}",
        "BRRlat": f"{s.loc['best_rule_recorded', 'lat_cov']:.4f}",
        "BRLx": f"{s.loc['best_rule_latent', 'x_ref']:.1f}", "BRLlat": f"{s.loc['best_rule_latent', 'lat_cov']:.4f}",
        "AttRecRec": f"{a.loc['ppo', 'x_ref']:.2f}", "AttLatRec": f"{a.loc['ppo_obslat', 'x_ref']:.2f}",
        "AttRecLat": f"{a.loc['ppo_rewlat', 'x_ref']:.2f}", "AttLatLat": f"{a.loc['ppo_oracle', 'x_ref']:.2f}",
        "PPOcoef": f"{e.loc['ppo', 'pooled_coef'] * 1e3:.1f}", "PPOt": f"{e.loc['ppo', 'pooled_t']:.1f}",
        "PPOseedsPos": e.loc["ppo", "seeds_positive"],
        "PPOcoefLo": f"{e.loc['ppo', 'seed_boot_lo'] * 1e3:.1f}", "PPOcoefHi": f"{e.loc['ppo', 'seed_boot_hi'] * 1e3:.1f}",
        "BRRcoef": f"{e.loc['best_rule_recorded', 'pooled_coef'] * 1e3:.1f}",
        "PtwoCoef": f"{e.loc['pen_p0.02', 'pooled_coef'] * 1e3:.1f}",
        "Ptwox": f"{s.loc['pen_p0.02', 'x_ref']:.2f}", "Ptwolat": f"{s.loc['pen_p0.02', 'lat_cov']:.4f}",
        "Pfourx": f"{s.loc['pen_p0.4', 'x_ref']:.2f}", "Pfourlat": f"{s.loc['pen_p0.4', 'lat_cov']:.4f}",
        "Rfivex": f"{s.loc['penref_w0.05', 'x_ref']:.2f}",
        "Ronefivex": f"{s.loc['penref_w0.15', 'x_ref']:.2f}", "Ronefivelat": f"{s.loc['penref_w0.15', 'lat_cov']:.4f}",
        "RonefiveLo": f"{s.loc['penref_w0.15', 'x_ref_lo']:.2f}", "RonefiveHi": f"{s.loc['penref_w0.15', 'x_ref_hi']:.2f}",
        "Rhalfx": f"{s.loc['penref_w0.5', 'x_ref']:.2f}",
        "RonefiveLatP": f"{c.loc['penref_w0.15|ppo|lat_cov', 'p_holm']:.3f}",
        "EfiveBase": f"{v('base', 'ppo').x_ref:.1f}", "EfiveBaseLo": f"{v('base', 'ppo').x_ref_lo:.1f}",
        "EfiveBaseHi": f"{v('base', 'ppo').x_ref_hi:.1f}",
        "EfivePerf": f"{v('perfect_obs', 'ppo').x_ref:.2f}", "EfiveConst": f"{v('const_detect', 'ppo').x_ref:.2f}",
        "EfiveConstLo": f"{v('const_detect', 'ppo').x_ref_lo:.2f}", "EfiveConstHi": f"{v('const_detect', 'ppo').x_ref_hi:.2f}",
        "EfiveFairMin": f"{e5[e5.kind == 'fair'].x_ref.min():.2f}", "EfiveFairMax": f"{e5[e5.kind == 'fair'].x_ref.max():.2f}",
        "PPObeta": f"{lr.loc['ppo', 'beta']:.1f}", "PPOw": f"{lr.loc['ppo', 'w']:+.2f}",
        "PPOmu": f"{lr.loc['ppo', 'mu']:.2f}",
        "NContrasts": str(len(ct)),
        "PPOsis": f"{s.loc['ppo', 'self_induced']:.2f}", "ORsis": f"{s.loc['oracle', 'self_induced']:.2f}",
        "EWMAsis": f"{s.loc['ewma_hotspot', 'self_induced']:.2f}",
        "PPOqfive": f"{qres['ppo'][4]:+.0f}", "PPOqtwothree": f"{min(qres['ppo'][1], qres['ppo'][2]):+.0f}",
        "ORqfive": f"{qres['oracle'][4]:+.0f}", "EWMAqfive": f"{qres['ewma_hotspot'][4]:+.0f}",
    }
    body = ["% Generated by scripts/make_paper_assets.py from results/final/ -- do not edit by hand."]
    for k, val in N.items():
        assert k.isalpha(), k
        body.append(f"\\newcommand{{\\n{k}}}{{{val}}}")
    Path("paper/numbers.tex").write_text("\n".join(body) + "\n")
    return N


def main():
    summ = pd.read_csv(FINAL / "e1_e4_summary.csv")
    per = pd.read_csv(FINAL / "per_training_seed.csv")
    e3t = pd.read_csv(FINAL / "e3_need_controlled.csv")
    att = pd.read_csv(FINAL / "attribution_2x2.csv")
    ct = pd.read_csv(FINAL / "contrasts.csv")
    e5 = pd.read_csv("results/e5_400k_e300n20.csv")
    rules = learned_rules()
    print(rules.groupby("family")[["beta", "w", "mu", "w_pos_share"]].mean().round(2).to_string())
    frames = load_frames()

    fig_arch()
    fig_e1(summ, per)
    fig_e2(frames, summ)
    fig_e3(frames, e3t)
    fig_e4(summ)
    fig_e5(e5)
    tables(summ, e3t, att, e5, ct, rules)
    need = long_run_need(frames[("uniform", -1)])
    groups = region.quantile_groups("hardship", 5)
    qres = {f: need_residual_by_quintile(frames, f, need, groups) * 100 for f in ["ppo", "oracle", "ewma_hotspot"]}
    N = numbers(summ, per, e3t, att, e5, ct, rules, qres)
    print(f"wrote 6 figures, {len(list(TAB.glob('*.tex')))} tables, {len(N)} numbers")


if __name__ == "__main__":
    main()

