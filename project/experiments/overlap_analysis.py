"""Overlap analysis: pairwise Jaccard heatmap and oracle bounds (Task 2.4).

Usage::

    python -m project.experiments.overlap_analysis
    python -m project.experiments.overlap_analysis --split valid
    python -m project.experiments.overlap_analysis --names EASE ItemKNN BPR NeuMF LightGCN
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project.analysis.overlap import (
    oracle_analysis,
    oracle_distribution,
    overlap_matrix,
    pairwise_overlap,
)
from project.metrics.evaluate import EvaluationContext
from project.utils.data_formats import (
    RECOMMENDATIONS_DIR,
    TOPK_CANDIDATES,
    TOPK_FINAL,
    load_recommendations,
)
from project.utils.paths import RESULTS_PROCESSED_DIR
from project.utils.report_assets import save_figure, save_latex_table


def _discover_names(split: str) -> list:
    """Find all recommendation lists available for a split."""
    pattern = f"*_{split}_top{TOPK_CANDIDATES}.csv"
    paths = sorted(RECOMMENDATIONS_DIR.glob(pattern))
    names = []
    for p in paths:
        name = p.stem.replace(f"_{split}_top{TOPK_CANDIDATES}", "")
        if name not in ("Random",):
            names.append(name)
    return names


def _plot_heatmap(mat: pd.DataFrame, split: str):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            val = mat.values[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, label="Mean Jaccard similarity", shrink=0.8)
    ax.set_title(f"Top-{TOPK_FINAL} pairwise overlap ({split})")
    plt.tight_layout()
    return fig


def _plot_oracle_comparison(model_ndcgs: dict, oracle_ndcg: float, split: str):
    names = sorted(model_ndcgs.keys(), key=lambda n: model_ndcgs[n])
    values = [model_ndcgs[n] for n in names]
    names.append("Oracle")
    values.append(oracle_ndcg)

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    colors = ["#4C72B0"] * (len(names) - 1) + ["#C44E52"]
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"NDCG@{TOPK_FINAL}")
    ax.set_title(f"Individual models vs. oracle ({split})")
    for i, v in enumerate(values):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=7)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Overlap and oracle analysis (Task 2.4)")
    parser.add_argument("--split", choices=["valid", "test", "both"], default="test")
    parser.add_argument("--names", nargs="+", default=None)
    args = parser.parse_args()
    splits = ("valid", "test") if args.split == "both" else (args.split,)

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Overlap analysis  split={split}")
        print(f"{'='*60}")

        names = args.names or _discover_names(split)
        print(f"  Models: {names}")

        lists = {n: load_recommendations(n, split) for n in names}
        ctx = EvaluationContext.for_split(split)

        print("\n  Computing pairwise overlap...")
        pw = pairwise_overlap(lists)
        pw.to_csv(RESULTS_PROCESSED_DIR / f"overlap_pairwise_{split}.csv", index=False)
        print(pw.to_string(index=False))

        mat = overlap_matrix(lists)
        fig = _plot_heatmap(mat, split)
        save_figure(fig, f"overlap_heatmap_{split}")
        plt.close(fig)
        print(f"\n  Heatmap saved.")

        print("\n  Computing oracle analysis...")
        ora = oracle_analysis(lists, ctx)
        oracle_ndcg = float(ora["oracle_result"].summary[f"ndcg@{TOPK_FINAL}"])
        print(f"  Oracle NDCG@{TOPK_FINAL}: {oracle_ndcg:.4f}")
        for name, ndcg in sorted(ora["model_ndcgs"].items(), key=lambda x: -x[1]):
            print(f"    {name}: {ndcg:.4f}")

        ora["picks"].to_csv(RESULTS_PROCESSED_DIR / f"oracle_picks_{split}.csv")

        summary_rows = [{"model": "Oracle", f"ndcg@{TOPK_FINAL}": oracle_ndcg}]
        for name, ndcg in ora["model_ndcgs"].items():
            summary_rows.append({"model": name, f"ndcg@{TOPK_FINAL}": ndcg})
        summary_df = pd.DataFrame(summary_rows).sort_values(
            f"ndcg@{TOPK_FINAL}", ascending=False
        ).reset_index(drop=True)
        summary_df.to_csv(RESULTS_PROCESSED_DIR / f"oracle_summary_{split}.csv", index=False)

        fig = _plot_oracle_comparison(ora["model_ndcgs"], oracle_ndcg, split)
        save_figure(fig, f"oracle_comparison_{split}")
        plt.close(fig)

        dist = oracle_distribution(ora["picks"], ctx.user_groups)
        dist.to_csv(RESULTS_PROCESSED_DIR / f"oracle_distribution_{split}.csv", index=False)
        print(f"\n  Oracle model picks by user group:")
        for dim in dist["dimension"].unique():
            print(f"\n    {dim}:")
            sub = dist[dist["dimension"] == dim].pivot_table(
                index="group", columns="model", values="share", fill_value=0
            )
            print(sub.to_string())

        save_latex_table(
            summary_df, f"overlap_oracle_{split}",
            float_format="%.4f", index=False,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
