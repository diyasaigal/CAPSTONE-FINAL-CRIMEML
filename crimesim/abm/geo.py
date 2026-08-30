"""Shared geography helpers: lat/lon <-> grid cell, and the CA lookup / real grid.

Everything that needs to turn a coordinate into a grid cell goes through
:func:`latlon_to_cell` so the ABM, the community-area lookup and the real-crime
evaluation grid are all on the same 20x20 (configurable) frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.config import DATA_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX


def latlon_to_cell(lat: np.ndarray, lon: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised. Returns (x, y) int cell indices; points outside the bbox are clipped."""
    xf = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    yf = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    x = np.clip((xf * grid).astype(int), 0, grid - 1)
    y = np.clip((yf * grid).astype(int), 0, grid - 1)
    return x, y


def _in_bbox(df: pd.DataFrame) -> pd.Series:
    return (
        df["Latitude"].between(LAT_MIN, LAT_MAX)
        & df["Longitude"].between(LON_MIN, LON_MAX)
    )


def build_ca_lookup(grid: int, crime_csv: str | Path | None = None) -> dict:
    """Build, from the real crime file:

    - ``ca_of_cell`` (grid, grid) int in 0..76 : modal community area of each cell
      (holes filled by nearest-populated-cell), for aggregating ABM crime to CAs.
    - ``real_grid`` (grid, grid) float : real violent-crime counts per cell,
      used by the static predictive-policing baseline and PAI/PEI.
    - ``real_ca`` (77,) float : real violent-crime counts per community area.
    - ``pop_weight_ca`` (77,) float : share of real crime per CA, a stand-in for
      population exposure when normalising enforcement.
    """
    path = Path(crime_csv) if crime_csv is not None else DATA_DIR / "processed_crime.csv"
    df = pd.read_csv(path, usecols=["Latitude", "Longitude", "Community Area"])
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[_in_bbox(df)].reset_index(drop=True)

    x, y = latlon_to_cell(df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), grid)
    df["cx"], df["cy"] = x, y

    real_grid = np.zeros((grid, grid), dtype=float)
    np.add.at(real_grid, (x, y), 1.0)

    # modal community area per cell
    ca_of_cell = np.full((grid, grid), -1, dtype=int)
    sub = df.dropna(subset=["Community Area"]).copy()
    sub["ca_pos"] = sub["Community Area"].astype(int) - 1
    mode = (
        sub.groupby(["cx", "cy"])["ca_pos"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    for (cx, cy), ca_pos in mode.items():
        ca_of_cell[cx, cy] = ca_pos

    # fill holes with the nearest populated cell
    holes = np.argwhere(ca_of_cell < 0)
    filled = np.argwhere(ca_of_cell >= 0)
    if len(holes) and len(filled):
        for hx, hy in holes:
            d2 = (filled[:, 0] - hx) ** 2 + (filled[:, 1] - hy) ** 2
            fx, fy = filled[d2.argmin()]
            ca_of_cell[hx, hy] = ca_of_cell[fx, fy]

    real_ca = np.zeros(77, dtype=float)
    vc = sub["ca_pos"].value_counts()
    real_ca[vc.index.to_numpy()] = vc.to_numpy()
    pop_weight_ca = real_ca / real_ca.sum()

    return {
        "grid": grid,
        "ca_of_cell": ca_of_cell,
        "real_grid": real_grid,
        "real_ca": real_ca,
        "pop_weight_ca": pop_weight_ca,
    }


def load_or_build_ca_lookup(grid: int) -> dict:
    cache = DATA_DIR / f"ca_lookup_g{grid}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    out = build_ca_lookup(grid)
    np.savez(
        cache,
        grid=np.array(grid),
        ca_of_cell=out["ca_of_cell"],
        real_grid=out["real_grid"],
        real_ca=out["real_ca"],
        pop_weight_ca=out["pop_weight_ca"],
    )
    return out
