import ast
import contextlib
import glob
import io
import json
import os
import random
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ml_validation
from crime_model_v5 import CrimeCity, Person

PAPER_FEATURES = [
    "sim_density",
    "log_sim",
    "sim_x",
    "sim_y",
    "nbr_sim",
    "nbr_sim_var",
    "x_norm",
    "y_norm",
    "dist_center",
    "q_nw",
    "q_ne",
    "q_sw",
    "hardship",
    "unemployment",
]


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def compute_feature_list_check() -> dict:
    real_df = pd.read_csv(os.path.join(ROOT, "processed_crime.csv"))
    real_grid = ml_validation.build_real_grid(real_df)
    sim_grid, risk_grid = ml_validation.run_abm(seed=42, gamma=0.2)
    feature_df = ml_validation.build_features(sim_grid, risk_grid)
    features = list(feature_df.columns)
    matches = features == PAPER_FEATURES
    payload = {
        "features_used_in_pipeline": features,
        "matches_paper_table": matches,
    }
    if not matches:
        expected_set = set(PAPER_FEATURES)
        actual_set = set(features)
        payload["extra_or_missing"] = {
            "missing": sorted(expected_set - actual_set),
            "extra": sorted(actual_set - expected_set),
        }
    return payload


def run_threshold_sweep(risk_threshold: float, interaction_threshold: float, seeds: int = 10) -> float:
    deltas = []
    for seed in range(seeds):
        low_total = _run_threshold_scenario("Low", risk_threshold, interaction_threshold, seed)
        high_total = _run_threshold_scenario("High", risk_threshold, interaction_threshold, seed)
        if low_total <= 0:
            pct_increase = 0.0
        else:
            pct_increase = ((high_total - low_total) / low_total) * 100.0
        deltas.append(float(pct_increase))

    arr = np.asarray(deltas, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci = (1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return round(mean, 3), round(ci, 3)


def _run_threshold_scenario(scenario_slug: str, risk_threshold: float, interaction_threshold: float, seed: int) -> int:
    original_interact = Person.interact

    def thresholded_interact(self):
        x, y = self.pos
        stats = self.model.loader.community_stats.get(self.community_area_id, {"hardship": 0.5})
        S = self.stress + stats["hardship"]
        T = self.model.risk_grid[x][y]
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        I = self.escalate(neighbors)
        hour = self.model.time % 24
        time_factor = 1.5 if (hour >= 20 or hour <= 2) else 1
        current = (S * T + self.model.gamma * I) * time_factor
        final = 0.8 * current + 0.2 * self.past_risk
        self.past_risk = final

        for n in neighbors:
            if isinstance(n, Person):
                combined = (self.stress + n.stress) / 2
                if combined > interaction_threshold and random.random() < 0.3:
                    self.crime = True
                    self.model.crime_map[x][y] += 1
                    break

        if final > risk_threshold:
            self.crime = True
            self.model.crime_map[x][y] += 1
            self.stress *= 0.5

        if self.pos == self.home:
            self.past_risk = 0

    Person.interact = thresholded_interact
    try:
        model = _make_threshold_model(scenario_slug, risk_threshold, interaction_threshold, seed)
        for _ in range(200):
            model.step()
        return int(model.crime_map.sum())
    finally:
        Person.interact = original_interact


def _make_threshold_model(scenario_slug: str, risk_threshold: float, interaction_threshold: float, seed: int) -> CrimeCity:
    scenario_map = {
        "Low": {"hardship_delta": 0.0, "unemp_delta": 0.0},
        "High": {"hardship_delta": 0.2, "unemp_delta": 0.2},
    }
    config = scenario_map[scenario_slug]

    random.seed(seed)
    np.random.seed(seed)
    with _quiet():
        model = CrimeCity(N=200, width=20, height=20, gamma=0.2)

    for ca in model.loader.community_stats:
        stats = model.loader.community_stats[ca]
        stats["hardship"] = min(1.0, stats["hardship"] + config["hardship_delta"])
        stats["unemployment"] = min(1.0, stats["unemployment"] + config["unemp_delta"])

    for agent in model.schedule.agents:
        if isinstance(agent, Person):
            stats = model.loader.community_stats.get(agent.community_area_id, {"unemployment": 0.15, "hardship": 0.5})
            agent.stress = 0.2 + stats["hardship"] + random.uniform(0, 0.2)

    model.crime_map = np.zeros((20, 20), dtype=float)
    return model


def compute_threshold_sensitivity() -> dict:
    output = {}
    for risk_threshold in [1.0, 1.2, 1.4]:
        mean, ci = run_threshold_sweep(risk_threshold=risk_threshold, interaction_threshold=0.7, seeds=10)
        output[f"risk_threshold_{risk_threshold}"] = {
            "pct_increase_mean": mean,
            "pct_increase_95ci_low": round(mean - ci, 3),
            "pct_increase_95ci_high": round(mean + ci, 3),
        }

    for interaction_threshold in [0.5, 0.7, 0.9]:
        mean, ci = run_threshold_sweep(risk_threshold=1.2, interaction_threshold=interaction_threshold, seeds=10)
        output[f"interaction_threshold_{interaction_threshold}"] = {
            "pct_increase_mean": mean,
            "pct_increase_95ci_low": round(mean - ci, 3),
            "pct_increase_95ci_high": round(mean + ci, 3),
        }
    return output


def summarize_test_coverage() -> dict:
    test_file_paths = sorted(
        p for p in glob.glob(os.path.join(ROOT, "**", "test*.py"), recursive=True)
        if ".venv" not in p.split(os.sep)
    )
    test_file_paths = [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in test_file_paths]

    covered_functions: list[str] = []
    test_case_count = 0

    for path in test_file_paths:
        full_path = os.path.join(ROOT, path)
        with open(full_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                test_case_count += 1
                lower = source.lower()
                if "interact" in lower or "escalate" in lower or "p = s*t + i" in lower or "s*t + i" in lower:
                    covered_functions.append(node.name)

    return {
        "test_file_paths": test_file_paths,
        "functions_covered": covered_functions,
        "num_test_cases": test_case_count,
    }


def main() -> None:
    result = {
        "feature_list_check": compute_feature_list_check(),
        "threshold_sensitivity": compute_threshold_sensitivity(),
        "unit_test_summary": summarize_test_coverage(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
