import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from crime_model_v5 import CrimeCity

GRID = 20
DEFAULT_SEED = 42
RESULTS_DIR = "results"


def set_seed(seed: int = DEFAULT_SEED) -> None:
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
    model = CrimeCity(N=300, width=GRID, height=GRID, gamma=gamma)
    for _ in range(400):
        model.step()
    return model.crime_map.copy(), model.risk_grid.copy()


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


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    print(f"{name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}, roc_auc={metrics['roc_auc']:.4f}")
    return metrics, y_prob, y_pred


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(DEFAULT_SEED)

    real_df = pd.read_csv("processed_crime.csv")
    real_grid = build_real_grid(real_df)
    sim_grid, risk_grid = run_abm(DEFAULT_SEED, gamma=0.2)
    feature_df = build_features(sim_grid, risk_grid)
    feature_names = [
        "x_norm", "y_norm", "dist_center", "q_nw", "q_ne", "q_sw",
        "sim_density", "log_sim", "nbr_sim", "nbr_sim_var",
        "hardship", "unemployment", "sim_x", "sim_y",
    ]

    X = feature_df[feature_names].to_numpy()
    y = label_hotspots(real_grid)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=DEFAULT_SEED
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=DEFAULT_SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=DEFAULT_SEED,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=DEFAULT_SEED),
    }

    scaler = StandardScaler()
    X_train_lr = scaler.fit_transform(X_train)
    X_test_lr = scaler.transform(X_test)

    metrics = {}
    metrics["logistic_regression"], _, _ = evaluate_model(
        "Logistic Regression",
        LogisticRegression(max_iter=2000, random_state=DEFAULT_SEED),
        X_train_lr,
        X_test_lr,
        y_train,
        y_test,
    )
    metrics["random_forest"], _, _ = evaluate_model(
        "Random Forest",
        models["random_forest"],
        X_train,
        X_test,
        y_train,
        y_test,
    )
    metrics["gradient_boosting"], _, _ = evaluate_model(
        "Gradient Boosting",
        models["gradient_boosting"],
        X_train,
        X_test,
        y_train,
        y_test,
    )

    with open(os.path.join(RESULTS_DIR, "ml_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rf_model = models["random_forest"]
    rf_model.fit(X_train, y_train)
    feature_importance = {
        "feature_names": feature_names,
        "feature_importances": rf_model.feature_importances_.astype(float).tolist(),
    }
    with open(os.path.join(RESULTS_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2)

    return {"results": metrics, "feature_names": feature_names}


if __name__ == "__main__":
    main()