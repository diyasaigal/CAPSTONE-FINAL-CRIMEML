# Run this inside your simulation codebase, where `low_grid` and `high_grid`
# are the 20x20 numpy arrays of simulated crime counts per cell for the
# Low (baseline) and High hardship scenarios (Table III/V in the paper).
# Saves figures/whatif_heatmaps.pdf, referenced by \includegraphics in the LaTeX.
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# low_grid = np.load("outputs/low_scenario_grid.npy")   # TODO: point at your real array
# high_grid = np.load("outputs/high_scenario_grid.npy") # TODO: point at your real array
#
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))
# vmax = max(low_grid.max(), high_grid.max())
# im0 = axes[0].imshow(low_grid, cmap="viridis", vmin=0, vmax=vmax)
# axes[0].set_title("Low hardship (baseline)")
# im1 = axes[1].imshow(high_grid, cmap="viridis", vmin=0, vmax=vmax)
# axes[1].set_title("High hardship (+0.20)")
# for ax in axes:
#     ax.set_xticks([]); ax.set_yticks([])
# fig.colorbar(im1, ax=axes, shrink=0.8, label="Simulated crime count per cell")
# plt.savefig("figures/whatif_heatmaps.pdf", bbox_inches="tight")
