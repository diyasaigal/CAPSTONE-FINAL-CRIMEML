# 🏙️ ABM Crime Simulation — Chicago
### Agent-Based Model + ML Validation Pipeline

> Simulates violent crime patterns across Chicago using agent-based modelling,
> validates against real crime data, runs socioeconomic what-if experiments,
> and trains hotspot classifiers with published output artifacts for a paper.

---

## 📁 Project Structure

```text
ABM/
├── crime_model_v5.py            # Main ABM; includes tunable gamma interaction multiplier
├── preprocess.py                # Data prep for processed_crime.csv and processed_census.csv
├── validate_pai_pei.py          # Honest hold-out hotspot validation (PAI / PEI)
├── whatif_experiment.py         # Three scenarios + multi-seed sensitivity analysis
├── ml_validation.py             # Independent ML model evaluation for LR / RF / GB
├── run_all_analysis.py          # Full pipeline runner
├── plot_feature_importance.py   # Legacy placeholder for manual feature plot generation
├── plot_whatif_heatmaps.py     # Legacy placeholder for manual heatmap generation
├── figures/
│   ├── plot_feature_importance.py
│   ├── plot_whatif_heatmaps.py
│   ├── feature_importance.pdf
│   └── whatif_heatmaps.pdf
├── outputs/
│   ├── low_scenario_grid.npy
│   └── high_scenario_grid.npy
├── results/
│   ├── ml_metrics.json
│   ├── sensitivity.csv
│   ├── feature_importance.json
│   └── summarize_for_paper.py
├── README.md
├── processed_crime.csv          # Filtered Chicago crime data used by the pipeline
├── processed_census.csv         # Community hardship + unemployment input data
├── crime_data.csv               # Raw source file (large; not tracked)
├── census_data.csv              # Raw source file (large; not tracked)
├── test_emergent_crime.py       # Sanity check for emergent crime dynamics
├── baseline_models.py           # Reference model experiments
└── validate_pai_pei.py          # Hotspot validation script
```

---

## 🚀 Run the pipeline

### 1) Preprocess data
```bash
python3 preprocess.py
```

### 2) Validate the ABM and generate paper outputs
```bash
python3 ml_validation.py
python3 whatif_experiment.py
python3 figures/plot_whatif_heatmaps.py
python3 figures/plot_feature_importance.py
```

### 3) Summarize results for the paper
```bash
python3 results/summarize_for_paper.py
```

This prints a single JSON object with the consolidated ML, sensitivity, and feature-importance summary.

---

## 🧠 Model and decision engine

The model is built in [crime_model_v5.py](crime_model_v5.py) and simulates a 20x20 grid representing Chicago. The core crime decision remains based on the stress-environment-interaction signal, with a tunable interaction multiplier `gamma`:

```python
current = (S * T + model.gamma * I) * time_factor
```

The current codebase supports the required paper-level sweep over `gamma in [0.1, 0.2, 0.3]` and stores the results in [results/sensitivity.csv](results/sensitivity.csv).

---

## 📌 Current measured paper outputs

### ML validation
The current pipeline independently trains and evaluates:
- Logistic Regression
- Random Forest
- Gradient Boosting

The results are saved to [results/ml_metrics.json](results/ml_metrics.json), using the schema:

```json
{
  "logistic_regression": {"accuracy": ..., "f1": ..., "roc_auc": ...},
  "random_forest": {"accuracy": ..., "f1": ..., "roc_auc": ...},
  "gradient_boosting": {"accuracy": ..., "f1": ..., "roc_auc": ...}
}
```

Current actual values generated from the codebase:
- Logistic Regression: accuracy 0.888, F1 0.640, ROC-AUC 0.923
- Random Forest: accuracy 0.888, F1 0.667, ROC-AUC 0.955
- Gradient Boosting: accuracy 0.875, F1 0.643, ROC-AUC 0.952

### Sensitivity analysis
The current what-if analysis runs the Low/Medium/High hardship scenarios across at least 10 seeds for each gamma value in `[0.1, 0.2, 0.3]`, and saves total crime counts to [results/sensitivity.csv](results/sensitivity.csv) with columns:

```text
scenario,gamma,seed,total_crimes
```

At `gamma = 0.2`, the code computes the % increase from Low to High hardship and prints the mean and 95% confidence interval to stdout. The current generated value is:
- mean % increase (Low -> High): 42.576%
- 95% CI: ±2.485%

### Feature importance
The Random Forest feature importances and corresponding feature names are saved to [results/feature_importance.json](results/feature_importance.json). The file is sorted by descending importances when summarized and is consumed by the plotting script.

### Scenario grids
The raw 20x20 crime-count grids for the low and high scenarios are saved as:
- [outputs/low_scenario_grid.npy](outputs/low_scenario_grid.npy)
- [outputs/high_scenario_grid.npy](outputs/high_scenario_grid.npy)

### Figures
The plotting scripts output publication-style PDFs to:
- [figures/whatif_heatmaps.pdf](figures/whatif_heatmaps.pdf)
- [figures/feature_importance.pdf](figures/feature_importance.pdf)

---

## 🔧 Paper summary script

The helper script [results/summarize_for_paper.py](results/summarize_for_paper.py) reads the generated JSON/CSV artifacts and prints one consolidated JSON object to stdout with the required schema. It validates file presence and required fields, and falls back to an `errors` array if any input is missing instead of inventing placeholders.

---

## 📝 Notes

- All ML models are trained independently and scored without reusing predictions across models.
- The Random Forest is the only model whose feature importance is exported to the paper artifact file.
- The generated PDFs are saved under the [figures](figures) directory and use Matplotlib without seaborn.
- The configuration is intentionally reproducible and every value in the paper summary comes from actual simulation and ML runs rather than hard-coded numbers.

---

*Built for the Chicago ABM crime simulation paper pipeline · 2026*
