"""Minimal OLS with heteroskedasticity-robust (HC1) standard errors.

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


def ols(y: np.ndarray, X: dict[str, np.ndarray], add_const: bool = True) -> OLSResult:
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
    # HC1 robust covariance
    S = (Xm * resid[:, None]).T @ (Xm * resid[:, None])
    cov = XtX_inv @ S @ XtX_inv * (n / max(n - k, 1))
    se = np.sqrt(np.diag(cov))
    tstat = beta / np.where(se > 0, se, np.nan)
    p = 2 * _sps.t.sf(np.abs(tstat), df=max(n - k, 1))
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
    return OLSResult(names, beta, se, tstat, p, float(r2), n)


def mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else 0.0, 0.0
    z = _sps.norm.ppf(0.5 + level / 2)
    return float(x.mean()), float(z * x.std(ddof=1) / np.sqrt(len(x)))
