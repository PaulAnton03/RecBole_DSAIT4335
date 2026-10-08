"""Tune the switching and mixed hybrids on the validation split (Task 1.5).

Usage::

    python -m project.experiments.tune_hybrids_d
    python -m project.experiments.tune_hybrids_d --only switching
    python -m project.experiments.tune_hybrids_d --only mixed
"""
from __future__ import annotations

import argparse
import json

from project.hybrids.tuning import LEARNED_MODELS, tune_mixed, tune_switching
from project.utils.data_formats import (
    EVAL_SPLITS,
    save_recommendations,
)
from project.utils.paths import RESULTS_PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser(description="Tune Person D's hybrids")
    parser.add_argument("--only", choices=["switching", "mixed"], default=None)
    args = parser.parse_args()

    tuning_dir = RESULTS_PROCESSED_DIR / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    best_configs = {}

    if args.only in (None, "switching"):
        print("=" * 60)
        print("Tuning SWITCHING hybrid")
        print("=" * 60)
        result = tune_switching(LEARNED_MODELS)
        result["trials"].to_csv(tuning_dir / "Switching_trials.csv", index=False)

        print(f"\n  Best dimension: {result['best_dimension']}")
        print(f"  Best assignment: {result['best_assignment']}")
        print(f"  Best NDCG@10: {result['best_ndcg']:.4f}")
        print(f"  Total trials: {len(result['trials'])}")
        print(f"\n  Top 10 configurations:")
        print(result["trials"].head(10).to_string(index=False))

        best_configs["switching"] = {
            "dimension": result["best_dimension"],
            "assignment": result["best_assignment"],
            "ndcg_valid": result["best_ndcg"],
        }

        print("\n  Saving best switching hybrid for both splits...")
        from project.hybrids.switching import switching_recommendations
        for split in EVAL_SPLITS:
            recs = switching_recommendations(
                result["best_assignment"], split, dimension=result["best_dimension"]
            )
            save_recommendations(recs, "Switching", split)
            print(f"    {split}: {len(recs)} rows")

    if args.only in (None, "mixed"):
        print("\n" + "=" * 60)
        print("Tuning MIXED hybrid")
        print("=" * 60)
        result = tune_mixed(LEARNED_MODELS)
        result["trials"].to_csv(tuning_dir / "Mixed_trials.csv", index=False)

        print(f"\n  Best method: {result['best_method']}")
        print(f"  Best models: {result['best_models']}")
        print(f"  Best rrf_k: {result['best_rrf_k']}")
        print(f"  Best NDCG@10: {result['best_ndcg']:.4f}")
        print(f"  Total trials: {len(result['trials'])}")
        print(f"\n  Top 10 configurations:")
        print(result["trials"].head(10).to_string(index=False))

        best_configs["mixed"] = {
            "method": result["best_method"],
            "models": result["best_models"],
            "rrf_k": result["best_rrf_k"],
            "ndcg_valid": result["best_ndcg"],
        }

        print("\n  Saving best mixed hybrid for both splits...")
        from project.hybrids.mixed import mixed_rrf, mixed_round_robin
        for split in EVAL_SPLITS:
            if result["best_method"] == "rrf":
                recs = mixed_rrf(result["best_models"], split, rrf_k=result["best_rrf_k"])
            else:
                recs = mixed_round_robin(result["best_models"], split)
            save_recommendations(recs, "Mixed", split)
            print(f"    {split}: {len(recs)} rows")

    if best_configs:
        out_path = RESULTS_PROCESSED_DIR / "hybrid_d_best.json"
        with open(out_path, "w") as f:
            json.dump(best_configs, f, indent=2)
        print(f"\nBest configs saved to {out_path}")

    print("\nDone. Run 'python -m project.experiments.evaluate_models' to evaluate.")


if __name__ == "__main__":
    main()
