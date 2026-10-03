"""Central configuration for the CrimeML feedback-loop / fair-RL project.

All dataclasses are plain and JSON/YAML-serialisable. Load overrides from a YAML
file with :func:`load_config`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# Canonical geography (Chicago). Used everywhere a lat/lon must become a grid
# cell so that the ABM, the CA lookup and the real-crime grid all agree.
# --------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 41.60, 42.05
LON_MIN, LON_MAX = -87.90, -87.52

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


@dataclass
class ABMConfig:
    """Parameters of the vectorised emergent-crime generator (ports crime_model_v5)."""
    grid: int = 20
    n_agents: int = 200
    gamma: float = 0.2                 # interaction-escalation multiplier (v5 default)
    risk_decay: float = 0.99           # risk_grid *= risk_decay each step
    risk_gain: float = 0.008           # risk_grid += risk_gain * occupancy  (v5: 0.01; re-tuned
                                       # in abm/calibration.py -- the mean-field port + 4 sub-steps
                                       # per shift shift the effective interaction magnitude)
    risk_memory: float = 0.2           # final = (1-m)*current + m*past_risk  (v5: 0.2)
    risk_trigger: float = 2.2          # final risk above this -> crime event  (v5: 1.2; re-tuned)
    interaction_trigger: float = 0.7   # mean pairwise stress above this -> escalation crime
    interaction_prob: float = 0.3      # per-encounter escalation probability (v5: 0.3)
    night_factor: float = 1.5          # time_factor for hours >=20 or <=2 (v5: 1.5)
    stress_base: float = 0.2           # stress = stress_base + hardship + U(0, stress_noise)
    stress_noise: float = 0.2
    crime_stress_relief: float = 0.5   # stress *= this after a threshold crime (v5: 0.5)
    patrol_stress_relief: float = 0.0  # v5's "police lower nearby stress" channel. Off by
                                       # default: patrol deters crime *directly* (DetectionConfig
                                       # .deterrence_*), which keeps the patrol->crime map linear
                                       # and interpretable. Can be turned on for an ablation.
    patrol_saturation: float = 1.5     # patrol-intensity at which full stress relief is reached
    patrol_reference: float = 20.0     # budget at which "proportional" allocation gives intensity 1
    p_work_local: float = 0.6          # prob an agent's workplace is in its home district
    # --- deviations from v5, off by default so the calibration gate matches exactly;
    #     the RL env (configs/*.yaml) turns these on for a stationary, targetable MDP ---
    stress_recovery: float = 0.0       # per-sub-step pull of stress back toward its baseline
    turnover_rate: float = 0.0         # per-shift prob an agent is resampled fresh
                                       # (residential mobility -> keeps the population stationary)
    steps_per_shift: int = 4           # ABM sub-steps per RL decision (one patrol shift)


@dataclass
class DetectionConfig:
    """Observation model: how latent crime becomes *recorded* crime, plus the
    physical deterrence effect of patrol on latent crime."""
    p_report: float = 0.30             # baseline reporting prob with zero patrol
    detect_gain: float = 0.55          # extra detection prob at full patrol saturation
    detect_saturation: float = 1.5     # patrol intensity at which detect_gain is reached
    deterrence_slope: float = 0.26     # latent-crime suppression per unit patrol intensity
    deterrence_cap: float = 0.68       # ...capped here (concentrating patrol still has limits)
                                       # det_frac(r) = min(cap, slope * intensity_r)


@dataclass
class EnvConfig:
    n_units: int = 20                  # patrol budget (patrol-hours) to allocate each shift
    min_share: float = 0.005           # floor on each district's budget share (no district is
                                       # ever fully abandoned; also bounds the disparity ratio)
    shifts_per_episode: int = 180      # ~2 months of 8h shifts
    warmup_shifts: int = 25            # shifts run under proportional patrol before the agent acts
    history: int = 4                   # #past shifts of recorded crime in the observation
    include_socioeconomic_obs: bool = True
    # what the trained policy optimises:
    #   "captured"     -- patrol-weighted recorded-crime hit rate (the deployed objective:
    #                     "be where crime is found"); drives the feedback loop
    #   "latent"       -- oracle: minimise true latent crime (not observable; upper bound)
    #   "recorded_neg" -- minimise total recorded crime (ablation)
    reward_on: str = "captured"
    reward_scale_coverage: float = 10.0   # coverage rewards are ~0.05; scale for PPO value fit
    # how the action maps to a deployment:
    #   "rule"   -- the policy outputs the parameters of an allocation rule
    #               (recency concentration, hardship weight, uniform mix). Low-dim,
    #               learnable, and a faithful model of a tuned predictive-policing
    #               system. This is the main setting.
    #   "direct" -- the policy outputs a raw logit per district (22-dim). Kept for
    #               an ablation; hard to explore in this noisy environment.
    action_mode: str = "rule"
    observe_latent: bool = False       # oracle only: put latent crime in the observation
    seed: int = 0


@dataclass
class FairnessConfig:
    protected_attr: str = "hardship"   # column used to split advantaged/disadvantaged
    n_quantiles: int = 5               # disparity = top-quantile vs bottom-quantile
    metric: str = "exposure_ratio"     # "exposure_ratio" | "gini" | "max_min"
    # penalty method (main): reward becomes  r - penalty * disparity_shift.
    # Sweeping `penalty` traces the efficiency-equity frontier by construction.
    penalty: float = 0.0
    # "linear": penalty * disparity_shift (pushes disparity toward 0, the original sweep)
    # "excess_log_ref": penalty * max(0, ln disparity_shift - ln ref_disparity), i.e. only
    #     disparity beyond a reference policy's (the GreedyOracle's) is penalised
    penalty_mode: str = "linear"
    ref_disparity: float = 0.0
    # Lagrangian method (alternative): dual ascent on lambda toward
    # E[max(0, disparity_cum - alpha)] <= cost_limit, i.e. disparity held at ~alpha.
    alpha: float = 2.0
    cost_limit: float = 0.0
    lambda_lr: float = 0.02
    lambda_init: float = 0.0
    lambda_max: float = 15.0


@dataclass
class RLConfig:
    algo: str = "ppo"                  # "ppo" | "lagrangian_ppo"
    total_timesteps: int = 400_000
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    n_envs: int = 8
    extractor: str = "mlp"            # "mlp" | "gnn"
    seed: int = 0


@dataclass
class Config:
    abm: ABMConfig = field(default_factory=ABMConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _merge(dc: Any, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return dc
    fields = {f.name: f for f in dataclasses.fields(dc)}
    kw = {}
    for k, v in overrides.items():
        if k not in fields:
            raise KeyError(f"unknown config key {k!r} for {type(dc).__name__}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            kw[k] = _merge(cur, v)
        else:
            kw[k] = v
    return dataclasses.replace(dc, **kw)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Build a :class:`Config`, optionally overlaying a YAML file then kwargs.

    ``overrides`` are nested dicts keyed by section, e.g.
    ``load_config(env={"n_units": 10}, abm={"grid": 30})``.
    """
    cfg = Config()
    if path is not None:
        data = _load_yaml_chain(Path(path))
        cfg = _merge(cfg, data)
    cfg = _merge(cfg, overrides)
    return cfg


def _deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a


def _load_yaml_chain(path: Path) -> dict:
    """Load a YAML config, honouring a top-level ``_extends: other.yaml`` key
    (resolved relative to this file) so runs can overlay ``configs/base.yaml``."""
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("_extends", None)
    if parent is None:
        return data
    base = _load_yaml_chain((path.parent / parent).resolve())
    return _deep_update(base, data)
