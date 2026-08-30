import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def main():
    os.makedirs("figures", exist_ok=True)
    low_grid = np.load("outputs/low_scenario_grid.npy")
    high_grid = np.load("outputs/high_scenario_grid.npy")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), constrained_layout=True)
    vmax = max(float(low_grid.max()), float(high_grid.max()))

    for ax, title, grid in [
        (axes[0], "Low hardship (baseline)", low_grid),
        (axes[1], "High hardship (+0.20)", high_grid),
    ]:
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Simulated crime count per cell")
    cbar.outline.set_linewidth(0.5)

    fig.savefig("figures/whatif_heatmaps.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
