"""Build & cache the grid-cell -> community-area lookup and the real-crime grid.

Run once (per grid size). Reads ``data/processed_crime.csv``; writes
``data/ca_lookup_g{grid}.npz``.
"""
import argparse

import numpy as np

from crimesim.abm.geo import build_ca_lookup
from crimesim.config import DATA_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=20)
    args = ap.parse_args()

    out = build_ca_lookup(args.grid)
    path = DATA_DIR / f"ca_lookup_g{args.grid}.npz"
    np.savez(
        path,
        grid=np.array(args.grid),
        ca_of_cell=out["ca_of_cell"],
        real_grid=out["real_grid"],
        real_ca=out["real_ca"],
        pop_weight_ca=out["pop_weight_ca"],
    )
    covered = (out["ca_of_cell"] >= 0).mean()
    print(f"wrote {path}")
    print(f"  grid {args.grid}x{args.grid}, cells with a CA: {covered:.1%}")
    print(f"  real violent crimes in bbox: {int(out['real_grid'].sum()):,}")
    print(f"  CAs with real crime: {(out['real_ca'] > 0).sum()}/77")
    top = np.argsort(out["real_ca"])[::-1][:5] + 1
    print(f"  top-5 CAs by crime: {top.tolist()}")


if __name__ == "__main__":
    main()
