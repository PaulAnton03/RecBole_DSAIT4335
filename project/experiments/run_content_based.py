"""Run the content-based model and export scores + recommendations.

Usage::

    python -m project.experiments.run_content_based            # both splits
    python -m project.experiments.run_content_based --split valid
"""
from __future__ import annotations

import argparse

import numpy as np

from project.hybrids.content_based import MODEL_NAME, build_content_scores
from project.utils.data_formats import (
    EVAL_SPLITS,
    TOPK_CANDIDATES,
    save_recommendations,
    save_scores,
    topk_from_scores,
)


def run(splits=EVAL_SPLITS):
    for split in splits:
        print(f"\n{'='*60}")
        print(f"ContentBased  split={split}")
        print(f"{'='*60}")

        sm = build_content_scores(split)
        save_scores(sm, MODEL_NAME, split)
        recs = topk_from_scores(sm, k=TOPK_CANDIDATES)
        save_recommendations(recs, MODEL_NAME, split)

        n_users = len(sm.user_id)
        n_items = len(sm.item_id)
        n_recs = len(recs)
        print(f"  users={n_users}  items={n_items}  recommendation rows={n_recs}")
        print(f"  score range: {sm.score[~np.isnan(sm.score)].min():.4f} .. "
              f"{sm.score[~np.isnan(sm.score)].max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the content-based model")
    parser.add_argument("--split", choices=["valid", "test", "both"], default="both")
    args = parser.parse_args()
    splits = EVAL_SPLITS if args.split == "both" else (args.split,)
    run(splits)
