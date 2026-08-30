"""
Master Demo Runner
===================
Runs the full ABM analysis pipeline in sequence:

  1. What-If Experiments (main causal result)
  2. ML Validation Layer (reality check)
  3. Optional PAI / PEI (legacy baseline)

Usage:
  python run_all_analysis.py
"""

import os
import time


def section(title: str) -> None:
    print("\n" + "╔" + "═" * 57 + "╗")
    print(f"║  {title:<55}║")
    print("╚" + "═" * 57 + "╝")


def main():
    print("\n" + "█" * 60)
    print("  ABM CRIME SIMULATION — FULL ANALYSIS PIPELINE")
    print("  Chicago Violent Crime | Agent-Based Model + ML")
    print("█" * 60)

    wall_start = time.time()
    summary = {}

    # ── TASK 1: WHAT-IF (MAIN STORY) ─────────────────────────────
    section("TASK 1 · WHAT-IF SOCIOECONOMIC EXPERIMENTS (MAIN)")
    from whatif_experiment import main as run_whatif
    t0 = time.time()
    task1 = run_whatif()
    summary["task1_time"] = time.time() - t0
    summary["scenario_totals"] = task1["totals"]

    # ── TASK 2: ML VALIDATION (REALITY CHECK) ────────────────────
    section("TASK 2 · MACHINE LEARNING VALIDATION (REALITY CHECK)")
    from ml_validation import main as run_ml
    t0 = time.time()
    task2 = run_ml()
    summary["task2_time"] = time.time() - t0

    rf_res = task2.get("results", {}).get("Random Forest", {})
    summary["rf_accuracy"] = rf_res.get("accuracy", 0)
    summary["rf_f1"] = rf_res.get("f1", 0)
    summary["rf_roc_auc"]  = rf_res.get("roc_auc", 0)

    # ── OPTIONAL: PAI / PEI (LEGACY COMPARISON) ──────────────────
    run_pai = os.getenv("RUN_PAI_PEI", "0") == "1"
    if run_pai:
        section("OPTIONAL · PAI / PEI VALIDATION PIPELINE")
        from validate_pai_pei import main as run_pai_fn
        t0 = time.time()
        task3 = run_pai_fn()
        summary["task3_time"] = time.time() - t0
        summary["PAI"] = task3["PAI"]
        summary["PEI"] = task3["PEI"]
        summary["hit_rate"] = task3["hit_rate"]

    # ── PRESENTATION SUMMARY ─────────────────────────────────────
    wall_total = time.time() - wall_start

    print("\n" + "█" * 60)
    print("  FINAL RESULTS — READY FOR PANEL")
    print("█" * 60)

    tots = summary["scenario_totals"]
    rf_acc = summary["rf_accuracy"]
    rf_f1 = summary["rf_f1"]
    rf_auc = summary["rf_roc_auc"]
    base = tots[0] if tots and tots[0] > 0 else 1
    med_delta = (tots[1] - tots[0]) / base * 100 if len(tots) > 1 else 0.0
    high_delta = (tots[2] - tots[0]) / base * 100 if len(tots) > 2 else 0.0

    print(f"""
    📌 TASK 1 · CAUSAL WHAT-IF (MAIN RESULT)
     Low    : {tots[0]:,}
     Medium : {tots[1]:,}
     High   : {tots[2]:,}
      Medium vs Low : {med_delta:+.1f}%
      High vs Low   : {high_delta:+.1f}%
      Main figure   : whatif_heatmaps.png

    📌 TASK 2 · ML REALITY CHECK
      Accuracy : {rf_acc*100:.1f}%
      F1 Score : {rf_f1*100:.1f}%
     ROC-AUC  : {rf_auc:.3f}
""")

    if run_pai:
        print(
            f"  Optional PAI/PEI (legacy): PAI={summary['PAI']:.2f}, "
            f"PEI={summary['PEI']:.2f}, Hit={summary['hit_rate']*100:.1f}%"
        )

    print(f"\n  Total runtime: {wall_total/60:.1f} minutes")

    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()