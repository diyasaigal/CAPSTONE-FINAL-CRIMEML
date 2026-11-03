"""Generator settings of the JASSS revision, as overlays on ``configs/jasss_base.yaml``.

Chosen at Gate 1 from ``scripts/phase1_generator.py`` (``results/phase1/``):

  contrast_low   hardship spread 0.2 -> latent crime per resident, top / bottom
                 hardship quintile, ~1.8x (1.8x remains with no hardship at all:
                 unemployment and geography alone give it)
  base           real hardship spread -> ~3.1x
  contrast_high  hardship also in the risk term, as in v5 -> ~4.3x (the
                 pre-revision world; hardship in stress alone tops out ~3.5x)
  agents_500/1000  same parameters with more agents; crime grows with N (no
                 re-tuning: holding crime fixed by raising the escalation trigger
                 pushed the contrast to 9-16x, a different world, not less noise)
"""
from __future__ import annotations

from crimesim.config import Config, load_config

BASE = "configs/jasss_base.yaml"

GENERATOR_VARIANTS: dict[str, dict] = {
    "base": {},
    "contrast_low": {"abm": {"hardship_scale": 0.2}},
    "contrast_high": {"abm": {"hardship_in_risk": True}},
    "agents_500": {"abm": {"n_agents": 500}},
    "agents_1000": {"abm": {"n_agents": 1000}},
}


def jasss_config(variant: str = "base", **overrides) -> Config:
    """``jasss_base.yaml`` + a generator variant + per-run ``overrides`` (nested dicts)."""
    over: dict = {}
    for src in (GENERATOR_VARIANTS[variant], overrides):
        for sec, vals in src.items():
            over.setdefault(sec, {}).update(vals)
    return load_config(BASE, **over)
