import json
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
    with open("results/feature_importance.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    importances = np.asarray(payload["feature_importances"], dtype=float)
    feature_names = payload["feature_names"]
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.barh(np.asarray(feature_names)[order], importances[order], color="#993556")
    ax.invert_yaxis()
    ax.set_xlabel("Random Forest feature importance")
    ax.set_ylabel("Feature")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig("figures/feature_importance.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
