"""User- and item-group analysis of every recommendation list (Task 2.5, Track B; reused for 3.4).

Reads every ``results/raw/recommendations/<Name>_<split>_top50.csv`` (or ``--names``), scores the
top 10 with our metrics and writes

    results/processed/group_definitions.json               thresholds and sizes of every group
    results/processed/user_group_metrics_<split>.csv       name x dimension x group x metric: mean, 95% CI, users
    results/processed/user_group_gaps_<split>.csv          max-min gap across groups (GRU extension) per metric
    results/processed/item_group_metrics_<split>.csv       name x item group: recall (+CI, support), exposure
    results/processed/popularity_alignment_<split>.csv     name x taste group: head/mid/tail/unseen shares, profile vs list
    results/processed/user_group_intersection_<split>.csv  activity x taste cells: users, NDCG, recall
    results/processed/user_group_pairs_<split>.csv         every two lists, per group: paired NDCG / recall difference + CI
    figures/generated/user_group_analysis.pdf              test: accuracy by user and item group
    figures/generated/popularity_alignment.pdf             test: popularity mix of profiles vs lists per taste group
    report/tables/generated/{group_definitions,user_group_results,item_group_results}.tex

Design decisions (e.g. a group-specific hybrid, Task 2.6) must use the ``valid`` tables; the
``test`` tables are for reporting. Groups and their thresholds: ``project/metrics/README.md``.

Usage (repository root)::

    python -m project.experiments.group_analysis                       # every list, valid + test
    python -m project.experiments.group_analysis --split valid --names EASE,WeightedHybrid
    python -m project.experiments.group_analysis --figure-names Random,Pop,EASE,BPR,WeightedHybrid
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project.analysis.group_analysis import analyse, model_order
from project.experiments.evaluate_models import available_names
from project.metrics import EvaluationContext, lookups
from project.models.registry import ALL_MODELS, DEFAULT_MODELS
from project.utils.data_formats import EVAL_SPLITS, TOPK_FINAL, load_recommendations, load_split
from project.utils.paths import RESULTS_PROCESSED_DIR
from project.utils.report_assets import save_figure, save_latex_table

# ordinal blue ramp (dataviz reference palette, steps 250 / 450 / 650; validated as an ordinal ramp):
# lighter = less active / more niche / less popular, darker = more
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
SHORT = {"low": "low", "medium": "med.", "high": "high", "niche": "niche", "mixed": "mixed", "mainstream": "main.",
         "head": "head", "mid": "mid", "tail": "tail", "unseen": "unseen"}


# ------------------------------------------------------------------------------------------ figures
def _style(ax, title):
    ax.set_title(title, fontsize=7.5, color=INK, loc="left", pad=13)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", labelsize=7, colors=INK_2, length=0)
    ax.tick_params(axis="x", colors=MUTED)


def _dot_panel(ax, values: pd.DataFrame, lows: pd.DataFrame, highs: pd.DataFrame, groups, title, legend_labels):
    """One row per list: a grey range line across the groups, one marker (+ 95% CI whisker) per group."""
    y = np.arange(len(values))[::-1]
    ax.hlines(y, values.min(axis=1), values.max(axis=1), color=AXIS, linewidth=1.2, zorder=1)
    for j, g in enumerate(groups):
        ax.hlines(y, lows[g], highs[g], color=RAMP[j], linewidth=1.0, alpha=0.9, zorder=2)
        ax.scatter(values[g], y, s=34, color=RAMP[j], edgecolor=SURFACE, linewidth=0.9, zorder=3,
                   label=legend_labels[j])
    ax.set_yticks(y, values.index)
    ax.set_xlim(left=0)
    _style(ax, title)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.04, 0.97), ncol=3, frameon=False, fontsize=6, handletextpad=0.1,
              columnspacing=0.6, borderaxespad=0.1, labelcolor=INK_2, markerscale=0.8)


def user_group_figure(user: pd.DataFrame, item: pd.DataFrame, names, k: int):
    def pivot(df, dim, metric):
        sub = df[(df.dimension == dim) & (df.metric == metric)]
        return [sub.pivot(index="name", columns="group", values=c).loc[names] for c in ("mean", "ci_low", "ci_high")]

    it = item[item.group.isin(["tail", "mid", "head"])]
    item_vals = [it.pivot(index="name", columns="group", values=c).loc[names]
                 for c in ("recall", "recall_ci_low", "recall_ci_high")]
    panels = [
        (pivot(user, "activity", f"ndcg@{k}"), lookups.ACTIVITY_GROUPS, f"(a) NDCG@{k}, user activity"),
        (pivot(user, "activity", f"recall@{k}"), lookups.ACTIVITY_GROUPS, f"(b) Recall@{k}, user activity"),
        (pivot(user, "taste", f"ndcg@{k}"), lookups.TASTE_GROUPS, f"(c) NDCG@{k}, user taste"),
        (item_vals, ("tail", "mid", "head"), f"(d) Recall@{k}, item group"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 0.2 * len(names) + 0.95), sharey=True)
    for ax, ((vals, lo, hi), groups, title) in zip(axes, panels):
        _dot_panel(ax, vals[list(groups)], lo, hi, list(groups), title, [SHORT[g] for g in groups])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    fig.tight_layout(w_pad=0.6)
    return fig


def alignment_figure(alignment: pd.DataFrame, names):
    """Stacked bars per taste group: popularity mix of the users' profiles and of each list."""
    parts = ["head", "mid", "tail"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 0.22 * (len(names) + 1) + 1.0), sharey=True)
    for ax, taste in zip(axes, lookups.TASTE_GROUPS):
        sub = alignment[alignment.taste == taste]
        profile = sub[sub.source == "profile"].iloc[0]
        rows = [("Profile (train)", profile)] + [(n, sub[(sub.source == "list") & (sub.name == n)].iloc[0]) for n in names]
        y = np.arange(len(rows))[::-1]
        left = np.zeros(len(rows))
        for j, part in enumerate(parts):  # head darkest, tail (incl. unseen) lightest
            w = np.array([r[part] + (r["unseen"] if part == "tail" else 0.0) for _, r in rows])
            ax.barh(y, w, left=left, height=0.62, color=RAMP[2 - j], edgecolor=SURFACE, linewidth=1.0,
                    label=part if part != "tail" else "tail (incl. unseen)")
            left += w
        ax.axhline(y[0] - 0.5, color=AXIS, linewidth=0.6)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        _style(ax, f"{taste} users (n={int(profile['users'])})")
        ax.title.set_position((0, 1))
    axes[0].set_yticks(y, [r for r, _ in rows])  # shared y axis: labels once, on the left
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7, labelcolor=INK_2,
               bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2)
    return fig


# ------------------------------------------------------------------------------------------- tables
def definitions_table(defs: dict) -> pd.DataFrame:
    rows = []
    for d in defs["users"]["activity"]:
        rows.append(("User activity", d["group"], f"{d['users']} users",
                     f"{d['min_score']:.0f}--{d['max_score']:.0f} training interactions"))
    for d in defs["users"]["taste"]:
        rows.append(("User taste", d["group"], f"{d['users']} users",
                     f"{d['min_score']:.0%}--{d['max_score']:.0%} of training interactions on head films".replace("%", "\\%")))
    for d in defs["items"]:
        rule = ("no training interaction" if d["max_count"] == 0 else
                f"{d['min_count']}--{d['max_count']} training interactions ({d['interaction_share']:.0%} of all)")
        rows.append(("Item popularity", d["group"], f"{d['items']} films", rule.replace("%", "\\%")))
    return pd.DataFrame(rows, columns=["Dimension", "Group", "Size", "Rule (training split)"])


def user_table(user: pd.DataFrame, names, k: int) -> pd.DataFrame:
    blocks = [(f"NDCG@{k}", "activity", f"ndcg@{k}"), (f"Recall@{k}", "activity", f"recall@{k}"),
              (f"NDCG@{k}", "taste", f"ndcg@{k}"), (f"UPD@{k}", "taste", f"upd@{k}")]
    cols = {}
    for label, dim, metric in blocks:
        sub = user[(user.dimension == dim) & (user.metric == metric)].pivot(index="name", columns="group", values="mean")
        for g in lookups.ACTIVITY_GROUPS if dim == "activity" else lookups.TASTE_GROUPS:
            cols[(f"{label} by {dim}", SHORT[g])] = sub.loc[names, g].values
    out = pd.DataFrame(cols, index=[n.replace("_", "\\_") for n in names])
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def item_table(item: pd.DataFrame, names, k: int) -> pd.DataFrame:
    cols = {}
    for label, col in ((f"Recall@{k}", "recall"), ("Share of slots", "slot_share")):
        sub = item.pivot(index="name", columns="group", values=col)
        for g in ("head", "mid", "tail"):
            cols[(label, g)] = sub.loc[names, g].values
    out = pd.DataFrame(cols, index=[n.replace("_", "\\_") for n in names])
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


# --------------------------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=EVAL_SPLITS + ("both",), default="both")
    parser.add_argument("--names", default=None, help="comma-separated list names (default: every exported list)")
    parser.add_argument("--figure-names", default=None,
                        help="lists in the figures (default: the shortlist of models plus every non-model list)")
    parser.add_argument("--k", type=int, default=TOPK_FINAL)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    splits = EVAL_SPLITS if args.split == "both" else (args.split,)
    results = {}
    for split in splits:
        ctx = EvaluationContext.for_split(split)
        names = args.names.split(",") if args.names else available_names(split)
        out = analyse({n: load_recommendations(n, split) for n in names}, ctx, args.k, args.n_boot)
        for key, file in (("user", "user_group_metrics"), ("gaps", "user_group_gaps"), ("item", "item_group_metrics"),
                          ("alignment", "popularity_alignment"), ("intersection", "user_group_intersection"),
                          ("pairs", "user_group_pairs")):
            path = RESULTS_PROCESSED_DIR / f"{file}_{split}.csv"
            out[key].to_csv(path, index=False, float_format="%.6g")
            print("wrote", path)
        results[split] = (out, names)

    gaps = results[splits[-1]][0]["gaps"]
    print(gaps[gaps.metric == f"ndcg@{args.k}"].round(3).to_string(index=False))

    if "test" in results and args.names is None:  # report assets come from the full test run
        out, names = results["test"]
        ctx = EvaluationContext.for_split("test")
        train = {u: set(items) for u, items in load_split("train").groupby("user_id")["item_id"]}
        defs = lookups.group_definitions(train, ctx.counts, ctx.item_groups, ctx.catalogue)  # training split only
        (RESULTS_PROCESSED_DIR / "group_definitions.json").write_text(json.dumps(defs, indent=2) + "\n")
        fig_names = (args.figure_names.split(",") if args.figure_names else
                     [n for n in DEFAULT_MODELS if n in names] + [n for n in names if n not in ALL_MODELS])
        print("wrote", save_figure(user_group_figure(out["user"], out["item"], fig_names, args.k), "user_group_analysis"))
        print("wrote", save_figure(alignment_figure(out["alignment"], fig_names), "popularity_alignment"))
        table_names = model_order(names, ALL_MODELS)
        print("wrote", save_latex_table(definitions_table(defs), "group_definitions", escape=False,
                                        column_format="llrl"))
        print("wrote", save_latex_table(user_table(out["user"], table_names, args.k), "user_group_results",
                                        index=True, escape=False, float_format="%.3f", multicolumn_format="c",
                                        column_format="l" + "r" * 12))
        print("wrote", save_latex_table(item_table(out["item"], table_names, args.k), "item_group_results",
                                        index=True, escape=False, float_format="%.3f", multicolumn_format="c",
                                        column_format="l" + "r" * 6))
        plt.close("all")


if __name__ == "__main__":
    main()
