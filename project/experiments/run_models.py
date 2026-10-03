"""Train, evaluate and export the individual models (Track A, Tasks 1.1 / 1.2 / 2.2).

For every model: train with RecBole on the frozen split, keep the best validation epoch,
report RecBole's validation and test metrics, and export

    results/raw/scores/<Model>_{valid,test}.npz                   all user-item scores
    results/raw/recommendations/<Model>_{valid,test}_top50.csv    candidate lists

The per-model metrics are appended to ``results/processed/recbole_metrics_<mode>.csv``.

Usage (repository root)::

    python -m project.experiments.run_models --mode quick                 # lecturer's hyper-parameters, shortlist
    python -m project.experiments.run_models --mode quick --models all    # all 11 models
    python -m project.experiments.run_models --mode tuned                 # after tune_models
    python -m project.experiments.run_models --mode quick --models BPR,Pop --epochs 2   # smoke test
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from project.models.pipeline import METRIC_COLUMNS, export_outputs, train_and_evaluate
from project.models.registry import MODES, resolve_models
from project.utils.data_formats import split_path
from project.utils.paths import RESULTS_PROCESSED_DIR


def metrics_csv(mode: str):
    return RESULTS_PROCESSED_DIR / f"recbole_metrics_{mode}.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, default="quick")
    parser.add_argument("--models", default="default", help='"default", "all" or comma-separated names')
    parser.add_argument("--epochs", type=int, default=None, help="override epochs (smoke tests only)")
    parser.add_argument("--no-export", action="store_true", help="skip writing score / recommendation files")
    args = parser.parse_args()

    if not split_path("test").exists():
        raise SystemExit("frozen split not found -- run `python -m project.experiments.export_split` first")
    overrides = {"epochs": args.epochs} if args.epochs else {}

    rows = []
    for spec in resolve_models(args.models):
        print(f"\n===== {spec.name} ({args.mode}) =====", flush=True)
        t0 = time.time()
        result, model, dataset, valid_data, test_data, config = train_and_evaluate(spec, args.mode, overrides)
        if not args.no_export:
            paths = export_outputs(spec, model, dataset, valid_data, test_data, config, result.test)
            print("exported:", ", ".join(str(p.relative_to(RESULTS_PROCESSED_DIR.parent.parent)) for p in paths.values()))
        for split, metrics in (("valid", result.best_valid), ("test", result.test)):
            rows.append({"model": spec.name, "family": spec.family, "mode": args.mode, "split": split,
                         **{m: metrics.get(m) for m in METRIC_COLUMNS},
                         "best_epoch": result.best_epoch, "train_seconds": round(result.train_seconds, 1),
                         "hyper_parameters": json.dumps(result.hyper_parameters, default=str)})
        print(f"{spec.name}: valid NDCG@10 {result.best_valid['ndcg@10']:.4f} | test NDCG@10 {result.test['ndcg@10']:.4f}"
              f" | best epoch {result.best_epoch} | {time.time() - t0:.0f}s", flush=True)

    new = pd.DataFrame(rows)
    path = metrics_csv(args.mode)
    if args.epochs:  # smoke test: do not pollute the tracked results
        path = path.with_name(path.stem + "_smoke.csv")
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["model"].isin(new["model"])]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    print(f"\nmetrics -> {path}")
    print(new[new.split == "test"][["model", "ndcg@10", "recall@10", "precision@10", "mrr@10", "hit@10", "map@10", "best_epoch", "train_seconds"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
