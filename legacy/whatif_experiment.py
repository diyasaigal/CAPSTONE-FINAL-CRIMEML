"""What-if socioeconomic experiment with sensitivity analysis and saved outputs."""

import csv
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from crime_model_v5 import CrimeCity, Person

GRID_SIZE = 20
SIM_STEPS = 200
N_AGENTS = 200
DEFAULT_SEED = 42
RESULTS_DIR = "results"
OUTPUTS_DIR = "outputs"

GAMMAS = [0.1, 0.2, 0.3]
SCENARIOS = [
    {"slug": "Low", "label": "Low Hardship", "hardship_delta": 0.0, "unemp_delta": 0.0, "color": "#2ec4b6"},
    {"slug": "Medium", "label": "Medium Hardship", "hardship_delta": 0.1, "unemp_delta": 0.1, "color": "#ff9f1c"},
    {"slug": "High", "label": "High Hardship", "hardship_delta": 0.2, "unemp_delta": 0.2, "color": "#e71d36"},
]


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_scenario_model(
    hardship_delta: float = 0.0,
    unemp_delta: float = 0.0,
    gamma: float = 0.2,
    n: int = N_AGENTS,
    grid_size: int = GRID_SIZE,
    seed: int = DEFAULT_SEED,
) -> CrimeCity:
    set_seed(seed)
    model = CrimeCity(N=n, width=grid_size, height=grid_size, gamma=gamma)

    for ca in model.loader.community_stats:
        stats = model.loader.community_stats[ca]
        stats["hardship"] = min(1.0, stats["hardship"] + hardship_delta)
        stats["unemployment"] = min(1.0, stats["unemployment"] + unemp_delta)

    for agent in model.schedule.agents:
        if isinstance(agent, Person):
            stats = model.loader.community_stats.get(
                agent.community_area_id,
                {"unemployment": 0.15, "hardship": 0.5},
            )
            agent.stress = 0.2 + stats["hardship"] + random.uniform(0, 0.2)

    model.crime_map = np.zeros((grid_size, grid_size), dtype=float)
    return model


def run_scenario(scenario: dict, steps: int = SIM_STEPS, gamma: float = 0.2, seed: int = DEFAULT_SEED):
    model = make_scenario_model(
        hardship_delta=scenario["hardship_delta"],
        unemp_delta=scenario["unemp_delta"],
        gamma=gamma,
        seed=seed,
    )
    for _ in range(steps):
        model.step()
    grid = model.crime_map.copy()
    total = int(grid.sum())
    return grid, total


def compute_pct_increase(low_total: float, high_total: float) -> float:
    if low_total <= 0:
        return 0.0
    return (high_total - low_total) / low_total * 100.0


def summarize_sensitivity() -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        for gamma in GAMMAS:
            for seed in range(10):
                grid, total = run_scenario(scenario, gamma=gamma, seed=seed)
                rows.append({
                    "scenario": scenario["slug"],
                    "gamma": float(gamma),
                    "seed": seed,
                    "total_crimes": int(total),
                })
    df = pd.DataFrame(rows, columns=["scenario", "gamma", "seed", "total_crimes"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "sensitivity.csv"), index=False)
    return df


def save_default_scenario_grids() -> tuple[np.ndarray, np.ndarray]:
    low_grid, _ = run_scenario(SCENARIOS[0], gamma=0.2, seed=DEFAULT_SEED)
    high_grid, _ = run_scenario(SCENARIOS[2], gamma=0.2, seed=DEFAULT_SEED)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUTS_DIR, "low_scenario_grid.npy"), low_grid)
    np.save(os.path.join(OUTPUTS_DIR, "high_scenario_grid.npy"), high_grid)
    return low_grid, high_grid


def compute_gamma020_low_to_high_ci(df: pd.DataFrame):
    gamma_df = df[(df["gamma"] == 0.2)].copy()
    low = gamma_df[gamma_df["scenario"] == "Low"]["total_crimes"].to_numpy()
    high = gamma_df[gamma_df["scenario"] == "High"]["total_crimes"].to_numpy()
    deltas = np.array([compute_pct_increase(l, h) for l, h in zip(low, high)])
    mean = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    return mean, ci, deltas


def plot_whatif(grids: list[np.ndarray], totals: list[int], scenarios: list[dict], output_path: str = "whatif_heatmaps.png"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    vmax = max(grid.max() for grid in grids) if grids else 1
    for ax, grid, scenario, total in zip(axes, grids, scenarios, totals):
        im = ax.imshow(grid.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(scenario["label"], pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"Total: {total:,}")
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.suptitle("What-if hardship scenarios")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    print("\nRunning multi-seed hardship sensitivity analysis...")
    sensitivity_df = summarize_sensitivity()
    mean_pct, ci, deltas = compute_gamma020_low_to_high_ci(sensitivity_df)
    print(f"Gamma=0.2 mean % increase (Low → High): {mean_pct:.4f}%")
    print(f"95% CI: ±{ci:.4f}%")

    low_grid, high_grid = save_default_scenario_grids()
    print(f"Saved low grid: {low_grid.shape}, high grid: {high_grid.shape}")

    grids = []
    totals = []
    for scenario in SCENARIOS:
        grid, total = run_scenario(scenario, gamma=0.2, seed=DEFAULT_SEED)
        grids.append(grid)
        totals.append(total)

    plot_whatif(grids, totals, SCENARIOS, output_path=os.path.join("figures", "whatif_heatmaps.pdf"))
    print("Sensitivity results saved to results/sensitivity.csv")
    return {
        "sensitivity": sensitivity_df.to_dict("records"),
        "mean_pct_increase_low_to_high_gamma_02": mean_pct,
        "ci_95": ci,
        "totals": totals,
    }


if __name__ == "__main__":
    main()
