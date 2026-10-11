"""Compare re-rank-then-combine vs combine-then-re-rank (Task 3.3).

The re-rankers are Track E's (``project/rerankers``), with Track E's lambda convention:
0 = the list's own order, 1 = the beyond-accuracy objective only.

Usage::

    python -m project.experiments.order_comparison
    python -m project.experiments.order_comparison --split valid --lambda 0.7
    python -m project.experiments.order_comparison --models EASE ItemKNN BPR NeuMF
    python -m project.experiments.order_comparison --objective Calibrated --lambda 0.4
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project.analysis.order_comparison import (
    combine_then_rerank,
    compare_orders,
    rerank_then_combine,
)
from project.hybrids.mixed import DEFAULT_MODELS, mixed_rrf
from project.metrics.evaluate import EvaluationContext
from project.rerankers import OBJECTIVES, list_reranker, make_objective
from project.utils.data_formats import (
    EVAL_SPLITS,
    TOPK_FINAL,
    save_recommendations,
)
from project.utils.paths import RESULTS_PROCESSED_DIR
from project.utils.report_assets import save_figure, save_latex_table


def _plot_comparison(comparison_df: pd.DataFrame, split: str, label: str):
    """Grouped bar chart: accuracy and beyond-accuracy metrics for both orders."""
    df = comparison_df.copy()
    metrics = df["metric"].tolist()
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_a = ax.bar(x - width / 2, df["rerank_then_combine"], width,
                    label="Re-rank then combine", color="#4C72B0")
    bars_b = ax.bar(x + width / 2, df["combine_then_rerank"], width,
                    label="Combine then re-rank", color="#DD8452")

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(f"Pipeline order comparison ({split}, {label})")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Order comparison (Task 3.3)")
    parser.add_argument("--split", choices=["valid", "test", "both"], default="test")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--objective", choices=list(OBJECTIVES), default="MMR",
                        help="Track E re-ranker (default: MMR)")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5,
                        help="re-ranker lambda, Track E convention (0 = list order, 1 = objective only)")
    args = parser.parse_args()
    splits = EVAL_SPLITS if args.split == "both" else (args.split,)
    models = args.models or DEFAULT_MODELS
    label = f"{args.objective}, lambda={args.lam}"
    suffix = "" if args.objective == "MMR" else f"_{args.objective}"  # MMR keeps the original file names

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Order comparison  split={split}  models={models}  {label}")
        print(f"{'='*60}")

        ctx = EvaluationContext.for_split(split)
        reranker_fn = list_reranker(make_objective(args.objective), args.lam, ctx)

        comparison_df, recs_a, recs_b = compare_orders(
            models, split, reranker_fn, mixed_rrf, ctx, k=TOPK_FINAL,
        )

        print(f"\n{comparison_df.to_string(index=False)}")

        save_recommendations(recs_a, f"OrderA_RerankThenCombine{suffix}", split)
        save_recommendations(recs_b, f"OrderB_CombineThenRerank{suffix}", split)

        comparison_df.to_csv(
            RESULTS_PROCESSED_DIR / f"order_comparison_{split}{suffix}.csv", index=False
        )

        fig = _plot_comparison(comparison_df, split, label)
        save_figure(fig, f"order_comparison_{split}{suffix}")
        plt.close(fig)

        save_latex_table(comparison_df, f"order_comparison_{split}{suffix}", float_format="%.4f")

        ndcg_a = comparison_df.loc[comparison_df["metric"] == f"ndcg@{TOPK_FINAL}",
                                    "rerank_then_combine"].values[0]
        ndcg_b = comparison_df.loc[comparison_df["metric"] == f"ndcg@{TOPK_FINAL}",
                                    "combine_then_rerank"].values[0]
        winner = "combine-then-rerank" if ndcg_b > ndcg_a else "rerank-then-combine"
        print(f"\n  NDCG@{TOPK_FINAL}: A={ndcg_a:.4f}  B={ndcg_b:.4f}  -> {winner} wins on accuracy")

    print("\nDone.")


if __name__ == "__main__":
    main()
