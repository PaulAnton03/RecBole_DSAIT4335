"""Impact of the re-rankers on user and item groups (Task 3.4, Track B).

The re-ranked lists are the ones Track E reports (``results/processed/reranker_results.csv``: base
list, objective and the lambda chosen on validation); a list missing from ``results/raw/recommendations``
is regenerated with Track E's ``rerank`` at that lambda (nothing is re-tuned). Groups are the frozen
Task 2.5 groups. For valid and test it writes

    results/processed/reranker_group_changes_<split>.csv  re-ranked - base per user group: paired mean difference
                                                          + 95% CI, base mean, relative change, for NDCG@10,
                                                          Recall@10 and the objective's target metric; ``robust``
                                                          = CI excludes 0 with the same sign on valid AND test
    results/processed/reranker_item_groups_<split>.csv    slot share, recall, coverage per item group, base and re-ranked
    figures/generated/reranker_group_impact.pdf           test: NDCG change and target-metric gain per user group
    report/tables/generated/reranker_item_groups.tex      test: item groups before / after re-ranking (appendix)

Usage (repository root)::

    python -m project.experiments.reranker_group_analysis
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from project.analysis.group_analysis import DIMENSIONS, N_BOOT, analyse_list, paired_differences
from project.metrics import METRIC_INFO, EvaluationContext
from project.rerankers import make_objective, rerank, reranked_name
from project.utils.data_formats import (EVAL_SPLITS, TOPK_FINAL, load_recommendations, recommendations_path,
                                        save_recommendations)
from project.utils.paths import RESULTS_PROCESSED_DIR
from project.utils.report_assets import save_figure, save_latex_table

CELLS = [("all", "all")] + [(d, g) for d, gs in DIMENSIONS.items() for g in gs]
SHORT = {"all": "all", "low": "low", "medium": "med.", "high": "high", "niche": "niche", "mixed": "mixed",
         "mainstream": "main.", "SLIMElastic": "SLIM"}
# dataviz reference palette: diverging red <-> blue around a grey midpoint; blue ramp for magnitude
DIVERGING = LinearSegmentedColormap.from_list("rb", ["#e34948", "#f0efec", "#256abf"])
SEQUENTIAL = LinearSegmentedColormap.from_list("b", ["#f0efec", "#256abf"])
INK, INK_2, MUTED, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#f0efec"


def reported_rerankings() -> pd.DataFrame:
    """Track E's reported re-rankings: base list, objective, lambda, list name, target metric."""
    res = pd.read_csv(RESULTS_PROCESSED_DIR / "reranker_results.csv").dropna(subset=["objective"])
    objs = [make_objective(o) for o in res["objective"]]
    return pd.DataFrame({"base": res["model"].values, "objective": [o.name for o in objs],
                         "lambda": res["lambda"].astype(float).values,
                         "name": [reranked_name(b, o) for b, o in zip(res["model"], objs)],
                         "target": [o.target_metric for o in objs]})


def ensure_lists(spec: pd.DataFrame, ctx: dict) -> None:
    """Regenerate a re-ranked list that is not in results/raw (it is git-ignored) at Track E's lambda."""
    for r in spec.itertuples():
        for split in EVAL_SPLITS:
            if not recommendations_path(r.name, split).exists():
                recs = rerank(load_recommendations(r.base, split), make_objective(r.objective), r.lambda_, ctx[split])
                print("regenerated", save_recommendations(recs, r.name, split), f"(lam = {r.lambda_:g})")


def group_changes(spec: pd.DataFrame, ctx: EvaluationContext, k: int, n_boot: int):
    """Paired re-ranked - base changes per user group, and item-group tables of every list."""
    names = list(dict.fromkeys(list(spec["base"]) + list(spec["name"])))
    parts = {n: analyse_list(load_recommendations(n, ctx.split), ctx, n, k, n_boot) for n in names}
    per_user = pd.concat([p["per_user"] for p in parts.values()])
    out = []
    for r in spec.itertuples():
        d = paired_differences(per_user, [], k, ["ndcg", "recall", r.target], n_boot, pairs=[(r.name, r.base)])
        out.append(d.assign(base=r.base, objective=r.objective, target=f"{r.target}@{k}"))
    ch = pd.concat(out, ignore_index=True)
    base_pu = per_user[per_user["name"].isin(spec["base"])]

    def base_mean(row):
        sub = base_pu[base_pu["name"] == row["b"]]
        sub = sub if row["dimension"] == "all" else sub[sub[row["dimension"]] == row["group"]]
        return sub[row["metric"]].mean()

    ch["base_mean"] = ch.apply(base_mean, axis=1)
    ch["rel_change"] = ch["diff"] / ch["base_mean"]
    items = pd.concat([parts[n]["item"] for n in names], ignore_index=True)
    return ch, items


def mark_robust(test: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """robust = the 95% interval excludes 0 with the same sign on both splits."""
    key = ["a", "metric", "dimension", "group"]
    v = valid.set_index(key)[["ci_low", "ci_high"]].reindex(test.set_index(key).index)
    up = (test["ci_low"].values > 0) & (v["ci_low"].values > 0)
    down = (test["ci_high"].values < 0) & (v["ci_high"].values < 0)
    return test.assign(robust=up | down)


def impact_figure(ch: pd.DataFrame, spec: pd.DataFrame, k: int):
    """(a) relative NDCG change per user group (shared diverging scale); (b) gain in the objective's own
    target metric (blue ramp scaled per row, value printed). Grey = not significant on both splits."""
    rows = [f"{SHORT.get(r.base, r.base)} + {r.objective} ({METRIC_INFO[r.target][0]})" for r in spec.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.25 * len(spec) + 1.05), sharey=True)
    ndcg = ch[ch.metric == f"ndcg@{k}"]
    vmax = np.nanmax(np.abs(ndcg["rel_change"])) or 1.0
    for panel, ax in enumerate(axes):
        for i, r in enumerate(spec.itertuples()):
            metric = f"ndcg@{k}" if panel == 0 else f"{r.target}@{k}"
            sub = ch[(ch.a == r.name) & (ch.metric == metric)].set_index(["dimension", "group"]).loc[CELLS]
            higher = METRIC_INFO[r.target][1]
            gain = sub["diff"] if (panel == 0 or higher) else -sub["diff"]
            row_max = np.nanmax(np.abs(gain)) or 1.0
            for j, cell in enumerate(CELLS):
                robust = bool(sub["robust"].iloc[j])
                if panel == 0:
                    v = sub["rel_change"].iloc[j]
                    color, text = DIVERGING(0.5 + 0.5 * v / vmax), f"{100 * v:+.0f}%"
                    dark = abs(v) / vmax > 0.6
                else:
                    v = gain.iloc[j]
                    color, text = SEQUENTIAL(max(v, 0) / row_max), f"{sub['diff'].iloc[j]:+.3f}"
                    dark = v / row_max > 0.6
                if not robust:
                    color, dark = NEUTRAL, False
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=1.2))
                ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=5.8,
                        color="white" if dark else (INK if robust else MUTED))
        ax.set_xlim(0, len(CELLS))
        ax.set_ylim(len(spec), 0)
        ax.set_xticks(np.arange(len(CELLS)) + 0.5, [SHORT[g] for _, g in CELLS], fontsize=6.5, color=INK_2)
        ax.xaxis.tick_top()
        ax.axvline(1, color="white", linewidth=3)
        ax.axvline(4, color="white", linewidth=3)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(["(a) NDCG@10 change (% of base)", "(b) target-metric change"][panel], fontsize=7.5,
                     color=INK, loc="left", pad=22)
    axes[0].set_yticks(np.arange(len(spec)) + 0.5, rows, fontsize=6.5, color=INK_2)
    for ax in axes:  # dimension labels above the column labels
        ax.text(2.5, -0.85, "activity", ha="center", fontsize=6, color=MUTED)
        ax.text(5.5, -0.85, "taste", ha="center", fontsize=6, color=MUTED)
    fig.tight_layout(w_pad=1.0)
    return fig


def item_table(items: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    cols, index = {}, []
    order = []
    for base in dict.fromkeys(spec["base"]):
        order.append((base, base))
        order += [(r.name, f"\\quad + {r.objective}") for r in spec[spec.base == base].itertuples()]
    for label, col in (("Share of slots", "slot_share"), ("Recall@10", "recall")):
        piv = items.pivot(index="name", columns="group", values=col)
        for g in ("head", "mid", "tail"):
            cols[(label, g)] = piv.loc[[n for n, _ in order], g].values
    cols[("Tail", "coverage")] = items.pivot(index="name", columns="group", values="coverage").loc[[n for n, _ in order], "tail"].values
    out = pd.DataFrame(cols, index=[lab for _, lab in order])
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=TOPK_FINAL)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    spec = reported_rerankings().rename(columns={"lambda": "lambda_"})
    ctx = {split: EvaluationContext.for_split(split) for split in EVAL_SPLITS}
    ensure_lists(spec, ctx)
    results = {split: group_changes(spec, ctx[split], args.k, args.n_boot) for split in EVAL_SPLITS}
    results["test"] = (mark_robust(results["test"][0], results["valid"][0]), results["test"][1])
    for split, (ch, items) in results.items():
        for df, file in ((ch, "reranker_group_changes"), (items, "reranker_item_groups")):
            path = RESULTS_PROCESSED_DIR / f"{file}_{split}.csv"
            df.to_csv(path, index=False, float_format="%.6g")
            print("wrote", path)

    ch, items = results["test"]
    print("wrote", save_figure(impact_figure(ch, spec, args.k), "reranker_group_impact"))
    print("wrote", save_latex_table(item_table(items, spec), "reranker_item_groups", index=True, escape=False,
                                    float_format="%.3f", multicolumn_format="c", column_format="l" + "r" * 7))
    plt.close("all")


if __name__ == "__main__":
    main()
