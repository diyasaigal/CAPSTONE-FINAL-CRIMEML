"""Observation model: how *latent* crime becomes *recorded* crime.

This is the mechanism that closes the feedback loop (cf. Ensign et al. 2018). An
area that receives more patrol has a higher chance of any given latent incident
being discovered/recorded, so patrolling an area makes it *look* worse in the
data even when the true rate is unchanged (or lowered by deterrence, which is
applied separately inside ``CrimeGenerator``).
"""
from __future__ import annotations

import numpy as np

from crimesim.config import DetectionConfig

MAX_P = 0.995


def detection_prob(intensity: np.ndarray, det: DetectionConfig) -> np.ndarray:
    """P(a latent incident in a CA is recorded), as a function of patrol intensity.

    ``intensity`` is units-in-CA / fair-share-units (1.0 == exactly proportional).
    """
    sat = np.clip(intensity / max(det.detect_saturation, 1e-9), 0.0, 1.0)
    p = det.p_report + det.detect_gain * sat
    return np.clip(p, 0.0, MAX_P)


def record(latent_region: np.ndarray, intensity: np.ndarray, det: DetectionConfig,
           rng: np.random.Generator) -> dict:
    p = detection_prob(intensity, det)
    recorded = rng.binomial(np.rint(latent_region).astype(np.int64), p).astype(float)
    return {"recorded": recorded, "p_detect": p}
