import csv
import json
import math
import os
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _round_float(value, digits=3):
    if value is None:
        return None
    return round(float(value), digits)


def _round_pct(value, digits=1):
    if value is None:
        return None
    return round(float(value), digits)


def _missing(errors, message):
    errors.append(message)


def main():
    errors = []
    result = {}

    ml_path = os.path.join(BASE_DIR, "ml_metrics.json")
    sensitivity_path = os.path.join(BASE_DIR, "sensitivity.csv")
    feature_path = os.path.join(BASE_DIR, "feature_importance.json")

    if not os.path.exists(ml_path):
        _missing(errors, f"Missing file: {ml_path}")
    if not os.path.exists(sensitivity_path):
        _missing(errors, f"Missing file: {sensitivity_path}")
    if not os.path.exists(feature_path):
        _missing(errors, f"Missing file: {feature_path}")

    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        return

    try:
        with open(ml_path, "r", encoding="utf-8") as fh:
            ml_data = json.load(fh)
    except Exception as exc:
        print(json.dumps({"errors": [f"Failed to read {ml_path}: {exc}"]}, indent=2))
        return

    try:
        with open(feature_path, "r", encoding="utf-8") as fh:
            feature_data = json.load(fh)
    except Exception as exc:
        print(json.dumps({"errors": [f"Failed to read {feature_path}: {exc}"]}, indent=2))
        return

    ml_required = ["logistic_regression", "random_forest", "gradient_boosting"]
    ml_metrics = {}
    for model_name in ml_required:
        if model_name not in ml_data:
            _missing(errors, f"Missing ml_metrics field: {model_name}")
            continue
        model_values = ml_data[model_name]
        if not isinstance(model_values, dict):
            _missing(errors, f"Invalid ml_metrics entry for {model_name}: expected object")
            continue
        f1_value = model_values.get("f1")
        if f1_value is None:
            f1_value = model_values.get("f1_score")
        if "accuracy" not in model_values or "roc_auc" not in model_values or f1_value is None:
            _missing(errors, f"Missing field(s) for {model_name}: accuracy, f1, roc_auc")
            continue
        ml_metrics[model_name] = {
            "accuracy": _round_float(model_values["accuracy"], 3),
            "f1": _round_float(f1_value, 3),
            "roc_auc": _round_float(model_values["roc_auc"], 3),
        }

    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        return

    sensitivity_rows = []
    with open(sensitivity_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sensitivity_rows.append(row)

    if not sensitivity_rows:
        print(json.dumps({"errors": [f"No rows found in {sensitivity_path}"]}, indent=2))
        return

    gamma_outputs = {}
    for gamma_label in ["0.1", "0.2", "0.3"]:
        selected = [r for r in sensitivity_rows if str(r.get("gamma")) == gamma_label]
        if not selected:
            _missing(errors, f"Missing sensitivity rows for gamma={gamma_label}")
            continue

        low_values = [float(r["total_crimes"]) for r in selected if str(r.get("scenario")) == "Low"]
        high_values = [float(r["total_crimes"]) for r in selected if str(r.get("scenario")) == "High"]
        if len(low_values) == 0 or len(high_values) == 0:
            _missing(errors, f"Missing scenario rows for gamma={gamma_label}")
            continue

        low_mean = statistics.mean(low_values)
        high_mean = statistics.mean(high_values)
        low_std = statistics.stdev(low_values) if len(low_values) > 1 else 0.0
        high_std = statistics.stdev(high_values) if len(high_values) > 1 else 0.0

        pct_increase = [((h - l) / l) * 100.0 for l, h in zip(low_values, high_values)]
        pct_mean = statistics.mean(pct_increase)
        pct_std = statistics.stdev(pct_increase) if len(pct_increase) > 1 else 0.0
        pct_margin = 1.96 * (pct_std / math.sqrt(len(pct_increase)))
        pct_ci_low = pct_mean - pct_margin
        pct_ci_high = pct_mean + pct_margin

        if gamma_label == "0.2":
            gamma_outputs["gamma_0.2"] = {
                "low_mean_crimes": _round_float(low_mean, 3),
                "low_std": _round_float(low_std, 3),
                "high_mean_crimes": _round_float(high_mean, 3),
                "high_std": _round_float(high_std, 3),
                "pct_increase_mean": _round_pct(pct_mean, 1),
                "pct_increase_95ci_low": _round_pct(pct_ci_low, 1),
                "pct_increase_95ci_high": _round_pct(pct_ci_high, 1),
                "n_seeds": int(len(low_values)),
            }
        else:
            gamma_outputs[f"gamma_{gamma_label}"] = {
                "pct_increase_mean": _round_pct(pct_mean, 1),
                "pct_increase_95ci_low": _round_pct(pct_ci_low, 1),
                "pct_increase_95ci_high": _round_pct(pct_ci_high, 1),
            }

    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        return

    feature_names = feature_data.get("feature_names")
    feature_importances = feature_data.get("feature_importances")
    if feature_names is None or feature_importances is None:
        print(json.dumps({"errors": ["Missing feature_importance keys: feature_names or feature_importances"]}, indent=2))
        return
    if len(feature_names) != len(feature_importances):
        print(json.dumps({"errors": ["feature_names and feature_importances length mismatch"]}, indent=2))
        return

    ranked_features = sorted(
        zip(feature_names, feature_importances),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    feature_summary = {name: _round_float(value, 3) for name, value in ranked_features}

    result = {
        "ml_metrics": ml_metrics,
        "sensitivity": gamma_outputs,
        "feature_importance": feature_summary,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
