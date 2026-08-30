"""Load the Chicago ACS / hardship table and expose per-community-area attributes.

The 77 community areas are the fairness units for the whole study. Index
convention: ``ca_pos`` 0..76 maps to community-area number ``ca_pos + 1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.config import DATA_DIR

N_CA = 77


@dataclass(frozen=True)
class Census:
    numbers: np.ndarray      # (77,) community-area numbers 1..77
    names: list[str]         # (77,) community-area names
    hardship: np.ndarray     # (77,) HARDSHIP INDEX / 100, in [0, 1]
    unemployment: np.ndarray # (77,) PERCENT AGED 16+ UNEMPLOYED / 100
    poverty: np.ndarray      # (77,) PERCENT HOUSEHOLDS BELOW POVERTY / 100
    income: np.ndarray       # (77,) PER CAPITA INCOME (USD)
    no_hs_diploma: np.ndarray

    def attribute(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def quantile_groups(self, attr: str, n_q: int) -> np.ndarray:
        """Return an int array (77,) with the quantile bin (0..n_q-1) of ``attr``."""
        x = self.attribute(attr)
        ranks = x.argsort().argsort()
        return (ranks * n_q // len(x)).astype(int)


def load_census(path: str | Path | None = None) -> Census:
    path = Path(path) if path is not None else DATA_DIR / "processed_census.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["Community Area Number"]).copy()
    df["Community Area Number"] = df["Community Area Number"].astype(int)
    df = df[df["Community Area Number"].between(1, 77)].sort_values("Community Area Number")
    assert len(df) == N_CA, f"expected {N_CA} community areas, got {len(df)}"

    # A couple of ACS cells are blank in the source; fill with column medians so the
    # simulator never sees NaN (documented in the paper's data section).
    hardship = (df["HARDSHIP INDEX"].astype(float) / 100.0)
    hardship = hardship.fillna(hardship.median()).to_numpy()

    return Census(
        numbers=df["Community Area Number"].to_numpy(),
        names=df["COMMUNITY AREA NAME"].str.strip().tolist(),
        hardship=hardship,
        unemployment=(df["PERCENT AGED 16+ UNEMPLOYED"].astype(float) / 100.0).to_numpy(),
        poverty=(df["PERCENT HOUSEHOLDS BELOW POVERTY"].astype(float) / 100.0).to_numpy(),
        income=df["PER CAPITA INCOME "].astype(float).to_numpy(),
        no_hs_diploma=(df["PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA"].astype(float) / 100.0).to_numpy(),
    )
