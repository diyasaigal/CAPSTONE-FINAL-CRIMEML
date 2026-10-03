"""Minimal OLS with heteroskedasticity-robust (HC1) or cluster-robust (CR1)
standard errors, plus a hierarchical bootstrap over training x eval seeds.

Kept dependency-free (no statsmodels). Returns coefficients, robust SEs, t-stats,
two-sided p-values and R^2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _sps


@dataclass
class OLSResult:
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    r2: float
    n: int

    def summary(self) -> str:
        rows = ["  term                 coef      se       t      p"]
        for nm, b, s, t, p in zip(self.names, self.beta, self.se, self.t, self.p):
            rows.append(f"  {nm:<18} {b:8.4f} {s:8.4f} {t:7.2f} {p:7.4f}")
        rows.append(f"  n={self.n}  R^2={self.r2:.3f}")
        return "\n".join(rows)

    def get(self, name: str) -> dict:
        i = self.names.index(name)
        return {"coef": float(self.beta[i]), "se": float(self.se[i]),
                "t": float(self.t[i]), "p": float(self.p[i])}


def ols(y: np.ndarray, X: dict[str, np.ndarray], add_const: bool = True,
        cluster: np.ndarray | None = None) -> OLSResult:
    """OLS with HC1 SEs, or CR1 cluster-robust SEs when ``cluster`` labels are given.

    Use clusters whenever rows are not independent (e.g. every region-shift of
    one episode): HC1 then understates the SEs by an order of magnitude.
    """
    names = list(X.keys())
    cols = [np.asarray(X[k], float) for k in names]
    if add_const:
        names = ["const"] + names
        cols = [np.ones_like(cols[0])] + cols
    Xm = np.column_stack(cols)
    y = np.asarray(y, float)
    n, k = Xm.shape

    XtX_inv = np.linalg.pinv(Xm.T @ Xm)
    beta = XtX_inv @ Xm.T @ y
    resid = y - Xm @ beta
    if cluster is None:
        # HC1 robust covariance
        S = (Xm * resid[:, None]).T @ (Xm * resid[:, None])
        cov = XtX_inv @ S @ XtX_inv * (n / max(n - k, 1))
        df = max(n - k, 1)
    else:
        # CR1: sum the score within each cluster, small-sample adjusted
        labels, idx = np.unique(np.asarray(cluster), return_inverse=True)
        G = len(labels)
        if G < 2:
            raise ValueError("cluster-robust SEs need at least 2 clusters")
        U = np.zeros((G, k))
        np.add.at(U, idx, Xm * resid[:, None])
        cov = XtX_inv @ (U.T @ U) @ XtX_inv * (G / (G - 1)) * ((n - 1) / max(n - k, 1))
        df = G - 1
    se = np.sqrt(np.diag(cov))
    tstat = beta / np.where(se > 0, se, np.nan)
    p = 2 * _sps.t.sf(np.abs(tstat), df=df)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
    return OLSResult(names, beta, se, tstat, p, float(r2), n)


def hierarchical_bootstrap(M: np.ndarray, n_boot: int = 4000, level: float = 0.95,
                           seed: int = 0) -> tuple[float, float, float]:
    """Mean of a (training seed x eval seed) matrix with a two-level percentile CI.

    Each replicate resamples rows (training seeds) and then columns (eval
    seeds) with replacement, so both sources of variability enter the CI.
    Columns are resampled jointly across rows, which keeps paired eval seeds
    paired. NaN cells are ignored. A single row reduces to an eval-seed bootstrap.
    """
    M = np.atleast_2d(np.asarray(M, float))
    rng = np.random.default_rng(seed)
    r, c = M.shape
    ri = rng.integers(0, r, (n_boot, r))
    ci = rng.integers(0, c, (n_boot, c))
    reps = np.array([np.nanmean(M[np.ix_(a, b)]) for a, b in zip(ri, ci)])
    a = (1 - level) / 2
    lo, hi = np.nanpercentile(reps, [100 * a, 100 * (1 - a)])
    return float(np.nanmean(M)), float(lo), float(hi)


def mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else 0.0, 0.0
    z = _sps.norm.ppf(0.5 + level / 2)
    return float(x.mean()), float(z * x.std(ddof=1) / np.sqrt(len(x)))
