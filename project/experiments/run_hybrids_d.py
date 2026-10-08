"""Run Person D's hybrids: content-based, switching, and mixed (Tasks 1.1, 1.4).

Usage::

    python -m project.experiments.run_hybrids_d               # both splits, all hybrids
    python -m project.experiments.run_hybrids_d --split valid
    python -m project.experiments.run_hybrids_d --only content_based switching
"""
from __future__ import annotations

import argparse

import numpy as np

from project.utils.data_formats import (
    EVAL_SPLITS,
    TOPK_CANDIDATES,
    save_recommendations,
    save_scores,
    topk_from_scores,
)

ALL_HYBRIDS = ("content_based", "switching", "mixed_rr", "mixed_rrf")


def _run_content_based(splits):
    from project.hybrids.content_based import MODEL_NAME, build_content_scores

    for split in splits:
        print(f"\n--- {MODEL_NAME}  split={split} ---")
        sm = build_content_scores(split)
        save_scores(sm, MODEL_NAME, split)
        recs = topk_from_scores(sm, k=TOPK_CANDIDATES)
        save_recommendations(recs, MODEL_NAME, split)
        valid_scores = sm.score[~np.isnan(sm.score)]
        print(f"  users={len(sm.user_id)}  recs={len(recs)}  "
              f"score range=[{valid_scores.min():.4f}, {valid_scores.max():.4f}]")


def _run_switching(splits):
    from project.hybrids.switching import (
        DEFAULT_ASSIGNMENT_ACTIVITY,
        switching_recommendations,
    )

    name = "Switching"
    for split in splits:
        print(f"\n--- {name}  split={split}  assignment={DEFAULT_ASSIGNMENT_ACTIVITY} ---")
        recs = switching_recommendations(DEFAULT_ASSIGNMENT_ACTIVITY, split, dimension="activity")
        save_recommendations(recs, name, split)
        print(f"  recommendation rows={len(recs)}")


def _run_mixed_rr(splits):
    from project.hybrids.mixed import DEFAULT_MODELS, mixed_round_robin

    name = "MixedRR"
    for split in splits:
        print(f"\n--- {name}  split={split}  models={DEFAULT_MODELS} ---")
        recs = mixed_round_robin(DEFAULT_MODELS, split)
        save_recommendations(recs, name, split)
        print(f"  recommendation rows={len(recs)}")


def _run_mixed_rrf(splits):
    from project.hybrids.mixed import DEFAULT_MODELS, DEFAULT_RRF_K, mixed_rrf

    name = "MixedRRF"
    for split in splits:
        print(f"\n--- {name}  split={split}  models={DEFAULT_MODELS}  rrf_k={DEFAULT_RRF_K} ---")
        recs = mixed_rrf(DEFAULT_MODELS, split)
        save_recommendations(recs, name, split)
        print(f"  recommendation rows={len(recs)}")


RUNNERS = {
    "content_based": _run_content_based,
    "switching": _run_switching,
    "mixed_rr": _run_mixed_rr,
    "mixed_rrf": _run_mixed_rrf,
}


def main():
    parser = argparse.ArgumentParser(description="Run Person D's hybrids")
    parser.add_argument("--split", choices=["valid", "test", "both"], default="both")
    parser.add_argument("--only", nargs="+", choices=ALL_HYBRIDS, default=None,
                        help="Run only these hybrids (default: all)")
    args = parser.parse_args()
    splits = EVAL_SPLITS if args.split == "both" else (args.split,)
    targets = args.only or ALL_HYBRIDS

    for key in targets:
        RUNNERS[key](splits)

    print("\nDone. Run 'python -m project.experiments.evaluate_models' to evaluate.")


if __name__ == "__main__":
    main()
