"""Spatial units for allocation and fairness.

The patrol-allocation unit is the **Chicago Police District** (22 of them) -- the
real operational unit and coarse enough to resolve on the simulation grid. Each
district carries socioeconomic covariates aggregated from the ACS community-area
table (``crimesim.abm.census``), crime-weighted, so "district hardship" reflects
the neighbourhoods the district actually polices.

``RegionInfo`` is the single object the generator, the environment and the
fairness meter all consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.abm.census import load_census
from crimesim.config import DATA_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from crimesim.abm.geo import latlon_to_cell

MIN_DISTRICT_CRIME = 500          # drop artefact districts (e.g. old #31)


@dataclass(frozen=True)
class RegionInfo:
    kind: str                     # "district"
    ids: np.ndarray               # (R,) original id labels (e.g. district numbers)
    n: int                        # R
    grid: int
    region_of_cell: np.ndarray    # (G,G) int in 0..R-1
    real_grid: np.ndarray         # (G,G) real violent-crime counts
    real_region: np.ndarray       # (R,) real violent-crime counts
    pop_weight: np.ndarray        # (R,) share of real crime (population-exposure proxy)
    hardship: np.ndarray          # (R,) crime-weighted ACS hardship in [0,1]
    poverty: np.ndarray           # (R,)
    unemployment: np.ndarray      # (R,)
    income: np.ndarray            # (R,) USD per capita
    onehot_cells: np.ndarray      # (G*G, R)

    def quantile_groups(self, attr: str, n_q: int) -> np.ndarray:
        x = getattr(self, attr)
        ranks = x.argsort().argsort()
        return (ranks * n_q // len(x)).astype(int)


def _in_bbox(df: pd.DataFrame) -> pd.Series:
    return df["Latitude"].between(LAT_MIN, LAT_MAX) & df["Longitude"].between(LON_MIN, LON_MAX)


def build_region_info(grid: int, region_col: str = "District",
                      crime_csv: str | Path | None = None) -> RegionInfo:
    path = Path(crime_csv) if crime_csv is not None else DATA_DIR / "processed_crime.csv"
    df = pd.read_csv(path, usecols=["Latitude", "Longitude", "District", "Community Area"])
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[_in_bbox(df)].reset_index(drop=True)

    keep = df[region_col].value_counts()
    keep = keep[keep >= MIN_DISTRICT_CRIME].index
    df = df[df[region_col].isin(keep)].copy()

    ids = np.sort(df[region_col].unique())
    id_to_pos = {v: i for i, v in enumerate(ids)}
    df["rpos"] = df[region_col].map(id_to_pos).astype(int)
    R = len(ids)

    x, y = latlon_to_cell(df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), grid)
    df["cx"], df["cy"] = x, y

    real_grid = np.zeros((grid, grid))
    np.add.at(real_grid, (x, y), 1.0)

    # modal region per cell (+ nearest-fill for empty cells)
    region_of_cell = np.full((grid, grid), -1, int)
    mode = df.groupby(["cx", "cy"])["rpos"].agg(lambda s: s.value_counts().idxmax())
    for (cx, cy), rp in mode.items():
        region_of_cell[cx, cy] = rp
    holes = np.argwhere(region_of_cell < 0)
    filled = np.argwhere(region_of_cell >= 0)
    for hx, hy in holes:
        d2 = (filled[:, 0] - hx) ** 2 + (filled[:, 1] - hy) ** 2
        fx, fy = filled[d2.argmin()]
        region_of_cell[hx, hy] = region_of_cell[fx, fy]
    assert (region_of_cell >= 0).all() and set(np.unique(region_of_cell)) == set(range(R))

    real_region = np.bincount(df["rpos"], minlength=R).astype(float)
    # Chicago Police districts are drawn for roughly balanced workload/population,
    # so we treat residential population as uniform across districts (a documented
    # simplification). This makes per-capita crime an *emergent* quantity that
    # varies with a district's socioeconomic profile rather than being baked in.
    pop_weight = np.full(R, 1.0 / R)

    # crime-weighted ACS covariates per region
    cen = load_census()
    ca = df["Community Area"].to_numpy()
    ok = np.isfinite(ca)
    ca_pos = np.where(ok, np.nan_to_num(ca, nan=1).astype(int) - 1, 0)

    def agg(vec):
        num = np.bincount(df["rpos"][ok], weights=vec[ca_pos][ok], minlength=R)
        den = np.bincount(df["rpos"][ok], minlength=R).astype(float)
        return num / np.clip(den, 1, None)

    return RegionInfo(
        kind="district", ids=ids, n=R, grid=grid,
        region_of_cell=region_of_cell, real_grid=real_grid, real_region=real_region,
        pop_weight=pop_weight,
        hardship=agg(cen.hardship), poverty=agg(cen.poverty),
        unemployment=agg(cen.unemployment), income=agg(cen.income),
        onehot_cells=np.eye(R)[region_of_cell.reshape(-1)],
    )


def load_region_info(grid: int, region_col: str = "District") -> RegionInfo:
    cache = DATA_DIR / f"region_{region_col.lower().replace(' ', '')}_g{grid}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return RegionInfo(kind=str(z["kind"]), ids=z["ids"], n=int(z["n"]), grid=int(z["grid"]),
                          region_of_cell=z["region_of_cell"], real_grid=z["real_grid"],
                          real_region=z["real_region"], pop_weight=z["pop_weight"],
                          hardship=z["hardship"], poverty=z["poverty"],
                          unemployment=z["unemployment"], income=z["income"],
                          onehot_cells=z["onehot_cells"])
    ri = build_region_info(grid, region_col)
    np.savez(cache, kind=ri.kind, ids=ri.ids, n=ri.n, grid=ri.grid,
             region_of_cell=ri.region_of_cell, real_grid=ri.real_grid,
             real_region=ri.real_region, pop_weight=ri.pop_weight, hardship=ri.hardship,
             poverty=ri.poverty, unemployment=ri.unemployment, income=ri.income,
             onehot_cells=ri.onehot_cells)
    return ri
