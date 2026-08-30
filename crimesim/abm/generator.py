"""``CrimeGenerator`` -- the vectorised emergent-crime simulator.

Environment for the RL study: :meth:`run_shift` advances the world one patrol
shift and returns the **latent** (ground-truth) violent crime per grid cell and
per region (Chicago police district). The RL policy never sees this directly --
only the *recorded* subset produced by ``crimesim.env.detection``.

Ports the dynamics of ``legacy/crime_model_v5.py`` (see ``dynamics.py`` for the
mean-field derivation). Agents are bound to a home region so the socioeconomic
geography is spatially coherent -- a prerequisite for a spatial fairness study,
and a fix to v5's position-independent random community-area assignment.
Physical patrol effects (deterrence, guardianship) live here; the
detection/reporting model lives in ``crimesim.env.detection``.
"""
from __future__ import annotations

import numpy as np

from crimesim.abm import dynamics as dyn
from crimesim.abm.regions import RegionInfo
from crimesim.config import ABMConfig, DetectionConfig


class CrimeGenerator:
    def __init__(self, cfg: ABMConfig, det: DetectionConfig, region: RegionInfo):
        self.cfg = cfg
        self.det = det
        self.region = region
        self.grid = region.grid
        self.R = region.n
        self.region_of_cell = region.region_of_cell
        w = np.where(region.pop_weight > 0, region.pop_weight, 1e-6)
        self.pop_weight = w / w.sum()
        self._onehot = region.onehot_cells                       # (G*G, R)

        G = self.grid
        flat = self.region_of_cell.reshape(-1)
        xs, ys = np.divmod(np.arange(G * G), G)
        order = np.argsort(flat, kind="stable")
        self._rcells = np.column_stack([xs[order], ys[order]])   # cells sorted by region
        self._rcount = np.bincount(flat, minlength=self.R)
        self._rstart = np.concatenate([[0], np.cumsum(self._rcount)[:-1]])
        self._home_p = self.pop_weight.copy()                    # uniform residential population
        self._flat_grid = G * G
        self.reset(np.random.default_rng(0))

    # ------------------------------------------------------------------
    def _cells_in(self, region_ids: np.ndarray, rng):
        """Vectorised: a uniformly random grid cell inside each given region."""
        region_ids = np.asarray(region_ids)
        off = (rng.random(len(region_ids)) * self._rcount[region_ids]).astype(int)
        cells = self._rcells[self._rstart[region_ids] + off]
        return cells[:, 0].copy(), cells[:, 1].copy()

    def reset(self, rng: np.random.Generator) -> None:
        c, G, N = self.cfg, self.grid, self.cfg.n_agents
        reg = self.region

        self.time = 0
        self.rid = rng.choice(self.R, size=N, p=self._home_p)     # home region
        self.employed = rng.random(N) < (1.0 - reg.unemployment[self.rid])
        self.stress0 = c.stress_base + reg.hardship[self.rid] + rng.uniform(0, c.stress_noise, N)
        self.stress = self.stress0.copy()

        self.hx, self.hy = self._cells_in(self.rid, rng)
        self.lx, self.ly = self._cells_in(self.rid, rng)
        work_r = np.where(rng.random(N) < c.p_work_local, self.rid,
                          rng.choice(self.R, size=N, p=self._home_p))
        self.wx, self.wy = self._cells_in(work_r, rng)
        self.ax, self.ay = self.hx.copy(), self.hy.copy()
        self.past_risk = np.zeros(N)

        self.risk_grid = np.full((G, G), 0.1)
        self._rng = rng

    # ------------------------------------------------------------------
    def patrol_intensity(self, patrol: np.ndarray | None) -> np.ndarray:
        """``intensity_r = deployment_r / (patrol_reference * pop_weight_r)``.

        Proportional allocation at exactly the reference budget -> intensity 1
        everywhere; a larger budget or concentration pushes it above 1.
        """
        if patrol is None:
            return np.zeros(self.R)
        ref = max(self.cfg.patrol_reference, 1e-9)
        return np.asarray(patrol, float) / (ref * self.pop_weight + 1e-9)

    @staticmethod
    def _sat(x, s):
        return np.clip(x / max(s, 1e-9), 0.0, 1.0)

    def _sub_step(self, intensity: np.ndarray) -> np.ndarray:
        c, G = self.cfg, self.grid
        self.time += 1
        hour = self.time % 24

        flat_pos = self.ax * G + self.ay
        self.risk_grid *= c.risk_decay
        occ = np.bincount(flat_pos, minlength=G * G).reshape(G, G)
        self.risk_grid += c.risk_gain * occ

        if 9 <= hour <= 17:
            tx = np.where(self.employed, self.wx, self.hx)
            ty = np.where(self.employed, self.wy, self.hy)
        elif 18 <= hour <= 23:
            tx, ty = self.lx.copy(), self.ly.copy()
        else:
            tx, ty = self.hx.copy(), self.hy.copy()
        self.ax, self.ay = dyn.moore_choice_towards(self.ax, self.ay, tx, ty, G)

        pos = self.ax * G + self.ay
        cnt = np.bincount(pos, minlength=G * G).reshape(G, G).astype(float)
        emp = np.bincount(pos[self.employed], minlength=G * G).reshape(G, G).astype(float)
        rs = dyn.ring_sum(np.stack([cnt, emp]))
        cell_n, cell_emp = rs[0], rs[1]

        n_events, self.past_risk, self.stress = dyn.crime_events(
            self._rng, stress=self.stress, hardship_ca=self.region.hardship, ca=self.rid,
            ax=self.ax, ay=self.ay, risk_grid=self.risk_grid, past_risk=self.past_risk,
            cell_n=cell_n, cell_emp=cell_emp, employed=self.employed, hour=hour, cfg=c,
        )
        latent = np.bincount(pos, weights=n_events.astype(float),
                             minlength=G * G).reshape(G, G)

        # deterrence: patrol suppresses latent crime, linear in intensity up to a cap
        det_frac = np.minimum(self.det.deterrence_cap, self.det.deterrence_slope * intensity)
        keep = np.clip(1.0 - det_frac[self.region_of_cell], 0.0, 1.0)
        latent = self._rng.binomial(latent.astype(np.int64), keep).astype(float)

        at_home = (self.ax == self.hx) & (self.ay == self.hy)
        self.past_risk[at_home] = 0.0

        # guardianship: patrol relieves agent stress (v5 Police mechanism)
        relief = 1.0 - c.patrol_stress_relief * self._sat(intensity, c.patrol_saturation)
        self.stress = self.stress * relief[self.rid]

        if c.stress_recovery > 0:
            self.stress += c.stress_recovery * (self.stress0 - self.stress)
        return latent

    def _turnover(self, rate: float) -> None:
        c, reg = self.cfg, self.region
        move = self._rng.random(c.n_agents) < rate
        k = int(move.sum())
        if k == 0:
            return
        idx = np.where(move)[0]
        r = self._rng.choice(self.R, size=k, p=self._home_p)
        self.rid[idx] = r
        self.employed[idx] = self._rng.random(k) < (1.0 - reg.unemployment[r])
        self.stress0[idx] = c.stress_base + reg.hardship[r] + self._rng.uniform(0, c.stress_noise, k)
        self.stress[idx] = self.stress0[idx]
        self.hx[idx], self.hy[idx] = self._cells_in(r, self._rng)
        self.lx[idx], self.ly[idx] = self._cells_in(r, self._rng)
        wr = self._rng.choice(self.R, size=k, p=self._home_p)
        self.wx[idx], self.wy[idx] = self._cells_in(wr, self._rng)
        self.ax[idx], self.ay[idx] = self.hx[idx], self.hy[idx]
        self.past_risk[idx] = 0.0

    # ------------------------------------------------------------------
    def run_shift(self, patrol: np.ndarray | None = None) -> dict:
        intensity = self.patrol_intensity(patrol)
        latent_cell = np.zeros((self.grid, self.grid))
        for _ in range(self.cfg.steps_per_shift):
            latent_cell += self._sub_step(intensity)
        if self.cfg.turnover_rate > 0:
            # residential mobility, applied once per shift (rate is per shift here)
            self._turnover(self.cfg.turnover_rate)
        latent_region = self._onehot.T @ latent_cell.reshape(-1)
        return {
            "latent_cell": latent_cell,
            "latent_region": latent_region,
            "intensity_region": intensity,
            "mean_stress": float(self.stress.mean()),
            "hour": self.time % 24,
        }
