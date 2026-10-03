"""Tune the individual models on validation NDCG@10 (Track A, Task 1.2).

Exhaustive grid per model from ``project/configs/hyper/<Model>.hyper``; writes the
winner to ``project/configs/models/<Model>.yaml``, all trials to
``results/processed/tuning/<Model>.csv`` and a one-line-per-model summary to
``results/processed/tuning_summary.csv``. Afterwards run
``python -m project.experiments.run_models --mode tuned`` to train, evaluate on test
and export the tuned models.

Usage (repository root)::

    python -m project.experiments.tune_models                      # shortlist (minus baselines)
    python -m project.experiments.tune_models --models BPR,EASE
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from project.models.registry import resolve_models
from project.models.tuning import tune_model
from project.utils.paths import RESULTS_PROCESSED_DIR

SUMMARY = RESULTS_PROCESSED_DIR / "tuning_summary.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="default", help='"default", "all" or comma-separated names')
    args = parser.parse_args()

    rows = []
    for spec in resolve_models(args.models):
        if not spec.tunable:
            print(f"skipping {spec.name} (baseline, nothing to tune)")
            continue
        t0 = time.time()
        row = tune_model(spec)
        row["tuning_seconds"] = round(time.time() - t0)
        rows.append(row)
        _merge_summary(pd.DataFrame([row]))
    print(f"\nsummary -> {SUMMARY}")
    print(pd.read_csv(SUMMARY).to_string(index=False))


def _merge_summary(new: pd.DataFrame):
    if SUMMARY.exists():
        old = pd.read_csv(SUMMARY)
        new = pd.concat([old[~old["model"].isin(new["model"])], new], ignore_index=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(SUMMARY, index=False)


if __name__ == "__main__":
    main()
