import builtins
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crime_model_v5 import CrimeCity

GRID = 20
DEFAULT_SEED = 42


def _quiet_print(enabled: bool):
    if enabled:
        builtins.print = _original_print
    else:
        builtins.print = lambda *args, **kwargs: None


_original_print = builtins.print


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)


def build_real_grid(df: pd.DataFrame, grid_size: int = GRID) -> np.ndarray:
    df = df.copy()
    df = df[
        (df["Latitude"] >= 41.6)
        & (df["Latitude"] <= 42.1)
        & (df["Longitude"] >= -87.95)
        & (df["Longitude"] <= -87.50)
    ].reset_index(drop=True)

    lat_min, lat_max = df["Latitude"].min(), df["Latitude"].max()
    lon_min, lon_max = df["Longitude"].min(), df["Longitude"].max()
    df["lat_norm"] = (df["Latitude"] - lat_min) / (lat_max - lat_min)
    df["lon_norm"] = (df["Longitude"] - lon_min) / (lon_max - lon_min)

    grid = np.zeros((grid_size, grid_size), dtype=float)
    x_idx = (df["lat_norm"] * (grid_size - 1)).astype(int).clip(0, grid_size - 1)
    y_idx = (df["lon_norm"] * (grid_size - 1)).astype(int).clip(0, grid_size - 1)
    for x, y in zip(x_idx, y_idx):
        grid[x, y] += 1.0
    return grid


def run_abm(seed: int = DEFAULT_SEED, gamma: float = 0.2):
    set_seed(seed)
    _quiet_print(False)
    try:
        model = CrimeCity(N=300, width=GRID, height=GRID, gamma=gamma)
        for _ in range(400):
            model.step()
        return model.crime_map.copy(), model.risk_grid.copy()
    finally:
        _quiet_print(True)


def build_features(sim_grid: np.ndarray, risk_grid: np.ndarray):
    community_stats = pd.read_csv("processed_census.csv")
    community_stats = community_stats.dropna(subset=["Community Area Number"]).copy()
    community_stats["Community Area Number"] = community_stats["Community Area Number"].astype(int)
    community_stats = community_stats.set_index("Community Area Number")
    community_values = community_stats[["unemployment", "hardship"]].to_dict("index")
    ca_keys = list(community_values.keys())

    rows = []
    for x in range(GRID):
        for y in range(GRID):
            density = float(sim_grid[x, y])
            patch = sim_grid[max(0, x - 1):min(GRID, x + 2), max(0, y - 1):min(GRID, y + 2)]
            x_norm = x / (GRID - 1)
            y_norm = y / (GRID - 1)
            dist_center = float(np.hypot(x_norm - 0.5, y_norm - 0.5))
            q_nw = float(x < GRID / 2 and y < GRID / 2)
            q_ne = float(x < GRID / 2 and y >= GRID / 2)
            q_sw = float(x >= GRID / 2 and y < GRID / 2)
            ca_id = ca_keys[(x * 17 + y * 13) % len(ca_keys)]
            stats = community_values[ca_id]
            rows.append({
                "x_norm": x_norm,
                "y_norm": y_norm,
                "dist_center": dist_center,
                "q_nw": q_nw,
                "q_ne": q_ne,
                "q_sw": q_sw,
                "sim_density": density,
                "log_sim": float(np.log1p(density)),
                "nbr_sim": float(patch.mean()),
                "nbr_sim_var": float(patch.var()),
                "hardship": float(stats["hardship"]),
                "unemployment": float(stats["unemployment"]),
                "sim_x": x_norm * density,
                "sim_y": y_norm * density,
            })
    return pd.DataFrame(rows)


def label_hotspots(real_grid: np.ndarray) -> np.ndarray:
    flat = real_grid.ravel()
    n_hotspots = int(0.2 * flat.size)
    order = np.argsort(flat)[::-1]
    labels = np.zeros(flat.size, dtype=int)
    labels[order[:n_hotspots]] = 1
    return labels


def evaluate_model(model, X_train, X_test, y_train, y_test, scaler=None):
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "f1": round(float(f1_score(y_test, y_pred)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 3),
    }


def load_expected_full_metrics():
    with open(os.path.join(os.path.dirname(__file__), "ml_metrics.json"), "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    return metrics


def main():
    errors = []
    set_seed(DEFAULT_SEED)

    try:
        real_df = pd.read_csv("processed_crime.csv")
    except Exception as exc:
        print(json.dumps({"errors": [f"Failed to read processed_crime.csv: {exc}"]}, indent=2))
        return

    real_grid = build_real_grid(real_df)
    sim_grid, risk_grid = run_abm(DEFAULT_SEED, gamma=0.2)
    feature_df = build_features(sim_grid, risk_grid)

    feature_names = [
        "x_norm", "y_norm", "dist_center", "q_nw", "q_ne", "q_sw",
        "sim_density", "log_sim", "nbr_sim", "nbr_sim_var",
        "hardship", "unemployment", "sim_x", "sim_y",
    ]

    required_drop = ["x_norm", "y_norm", "dist_center"]
    missing_required = [name for name in required_drop if name not in feature_names]
    if missing_required:
        print(json.dumps({
            "errors": [
                f"Feature mismatch: expected {required_drop} in the pipeline matrix, but the actual matrix contains {feature_names}."
            ]
        }, indent=2))
        return

    X = feature_df[feature_names].to_numpy()
    y = label_hotspots(real_grid)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=DEFAULT_SEED
    )

    expected = load_expected_full_metrics()
    full = {}
    no_coords = {}

    for model_name, factory in {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=DEFAULT_SEED),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=DEFAULT_SEED,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=DEFAULT_SEED),
    }.items():
        model = factory()
        if model_name == "logistic_regression":
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test, scaler=StandardScaler())
        else:
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test, scaler=None)
        full[model_name] = metrics

        if model_name == "logistic_regression":
            no_coords_model = LogisticRegression(max_iter=2000, random_state=DEFAULT_SEED)
            coord_columns = ["x_norm", "y_norm", "dist_center"]
            present = [c for c in coord_columns if c in feature_names]
            if present:
                X_train_no = np.delete(X_train, [feature_names.index(c) for c in present], axis=1)
                X_test_no = np.delete(X_test, [feature_names.index(c) for c in present], axis=1)
                no_coords[model_name] = evaluate_model(no_coords_model, X_train_no, X_test_no, y_train, y_test, scaler=StandardScaler())
            else:
                errors.append("Missing expected coordinate features for logistic_regression")
        elif model_name == "random_forest":
            no_coords_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                random_state=DEFAULT_SEED,
            )
            coord_columns = ["x_norm", "y_norm", "dist_center"]
            present = [c for c in coord_columns if c in feature_names]
            if present:
                X_train_no = np.delete(X_train, [feature_names.index(c) for c in present], axis=1)
                X_test_no = np.delete(X_test, [feature_names.index(c) for c in present], axis=1)
                no_coords[model_name] = evaluate_model(no_coords_model, X_train_no, X_test_no, y_train, y_test, scaler=None)
            else:
                errors.append("Missing expected coordinate features for random_forest")
        else:
            no_coords_model = GradientBoostingClassifier(random_state=DEFAULT_SEED)
            coord_columns = ["x_norm", "y_norm", "dist_center"]
            present = [c for c in coord_columns if c in feature_names]
            if present:
                X_train_no = np.delete(X_train, [feature_names.index(c) for c in present], axis=1)
                X_test_no = np.delete(X_test, [feature_names.index(c) for c in present], axis=1)
                no_coords[model_name] = evaluate_model(no_coords_model, X_train_no, X_test_no, y_train, y_test, scaler=None)
            else:
                errors.append("Missing expected coordinate features for gradient_boosting")

    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        return

    for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        expected_model = expected.get(model_name, {})
        for metric_name in ["accuracy", "f1", "roc_auc"]:
            if metric_name in expected_model:
                expected_value = float(expected_model[metric_name])
                actual_value = float(full[model_name][metric_name])
                if round(actual_value, 3) != round(expected_value, 3):
                    errors.append(f"Mismatch for {model_name}.{metric_name}: expected {expected_value}, got {actual_value}")

    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        return

    result = {
        "full": full,
        "no_coords": no_coords,
        "dropped_features": ["x_norm", "y_norm", "dist_center"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
