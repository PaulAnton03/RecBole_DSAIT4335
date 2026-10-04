"""Evaluate recommendation lists with our own metrics (Task 2.1, Track B).

Reads every ``results/raw/recommendations/<Name>_<split>_top50.csv`` (or ``--names``), scores
the top 10 with ``project.metrics.evaluate`` and writes

    results/processed/metrics_<split>.csv               one row per list: every metric + user counts
    results/raw/metrics/<Name>_<split>_per_user.csv     per-user values (input of the group analyses, Task 2.5)
    results/processed/metric_validation.csv             our metrics vs. RecBole on the very same lists
    report/tables/generated/beyond_accuracy_results.tex beyond-accuracy metrics of every list on test
    report/tables/generated/metric_validation.tex       largest |ours - RecBole| per metric (appendix)

Validation (nothing reported in the paper is computed by RecBole): the accuracy metrics are
compared with the numbers RecBole printed for the same checkpoint (``recbole_metrics_tuned.csv``,
falling back to ``recbole_metrics_quick.csv``), and coverage / Gini / entropy / average popularity
with RecBole's own metric code applied to our lists. ILD, novelty, serendipity, miscalibration,
UPD, GRU and the tail share have no RecBole counterpart; they are covered by the hand-computed
unit tests in ``project/tests/test_metrics.py``.

Usage (repository root)::

    python -m project.experiments.evaluate_models                         # all lists, valid + test
    python -m project.experiments.evaluate_models --split test --names EASE,Pop
    python -m project.experiments.evaluate_models --min-rating 4          # relevance = rating >= 4 (appendix)
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd

from project.metrics import METRIC_INFO, EvaluationContext, evaluate, lookups
from project.metrics.evaluate import index_matrix
from project.models.registry import ALL_MODELS
from project.utils.data_formats import (EVAL_SPLITS, RECOMMENDATIONS_DIR, TOPK_CANDIDATES, TOPK_FINAL,
                                        load_recommendations)
from project.utils.paths import RESULTS_PROCESSED_DIR, RESULTS_RAW_DIR
from project.utils.report_assets import save_latex_table

PER_USER_DIR = RESULTS_RAW_DIR / "metrics"
ACCURACY = ["ndcg", "recall", "precision", "mrr", "hit", "map"]  # the metrics RecBole reports for every run
TABLE_METRICS = ["ild", "novelty", "serendipity", "avgpop", "tailshare", "miscalibration", "upd", "gru_ndcg",
                 "coverage", "gini"]
SHORT_LABELS = {"novelty": "Nov.", "serendipity": "Ser.", "tailshare": "Tail", "coverage": "Cov."}  # fit A4


def available_names(split: str) -> List[str]:
    suffix = f"_{split}_top{TOPK_CANDIDATES}.csv"
    names = [p.name[: -len(suffix)] for p in RECOMMENDATIONS_DIR.glob(f"*{suffix}")]
    order = {m: i for i, m in enumerate(ALL_MODELS)}  # models in registry order, then hybrids / re-rankers
    return sorted(names, key=lambda n: (order.get(n, len(order)), n))


def recbole_reported() -> pd.DataFrame:
    """RecBole's own numbers per (model, split): tuned run, else the quick run (baselines)."""
    frames = [pd.read_csv(p) for mode in ("tuned", "quick")
              if (p := RESULTS_PROCESSED_DIR / f"recbole_metrics_{mode}.csv").exists()]
    if not frames:
        return pd.DataFrame(columns=["model", "split"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["model", "split"], keep="first")


def recbole_metric_code(idx: np.ndarray, ctx: EvaluationContext, k: int) -> dict:
    """Coverage, Gini, entropy and average popularity computed by RecBole's metric classes on our lists."""
    import recbole.evaluator.metrics as rb

    full = idx[(idx >= 0).all(axis=1)]  # RecBole's code assumes complete lists
    cov, gini, ent, pop = (rb.ItemCoverage.__new__(rb.ItemCoverage), rb.GiniIndex.__new__(rb.GiniIndex),
                           rb.ShannonEntropy.__new__(rb.ShannonEntropy), rb.AveragePopularity.__new__(rb.AveragePopularity))
    avg_count = pop.metric_info(pop.get_pop(full, dict(enumerate(ctx.counts))).astype(float))[:, -1].mean()
    return {
        f"coverage@{k}": cov.get_coverage(full, ctx.n_items),
        f"gini@{k}": gini.get_gini(full, ctx.n_items),
        # RecBole divides the entropy by the number of distinct recommended items; we normalise by ln |I|
        f"entropy@{k}": ent.get_entropy(full) * len(np.unique(full)) / np.log(ctx.n_items),
        f"avgpop@{k}": avg_count / ctx.n_history_users,
    }


def validation_rows(name, split, summary, idx, ctx, k, reported: pd.DataFrame) -> list:
    rows = []
    rep = reported[(reported["model"] == name) & (reported["split"] == split)]
    if len(rep):
        for m in ACCURACY:
            rows.append({"name": name, "split": split, "metric": f"{m}@{k}", "ours": summary[f"{m}@{k}"],
                         "recbole": float(rep.iloc[0][f"{m}@{k}"]), "source": "RecBole run (4 decimals)"})
    for metric, value in recbole_metric_code(idx, ctx, k).items():
        rows.append({"name": name, "split": split, "metric": metric, "ours": summary[metric], "recbole": value,
                     "source": "RecBole metric code"})
    return rows


def results_table(summary: pd.DataFrame, k: int) -> pd.DataFrame:
    test = summary[summary["split"] == "test"].set_index("name")
    out = pd.DataFrame({"Model": test.index.str.replace("_", r"\_")})
    for m in TABLE_METRICS:
        label, higher = SHORT_LABELS.get(m, METRIC_INFO[m][0]), METRIC_INFO[m][1]
        arrow = r"$\uparrow$" if higher else r"$\downarrow$"
        out[f"{label} {arrow}"] = test[f"{m}@{k}"].values
    return out


def validation_table(val: pd.DataFrame) -> pd.DataFrame:
    """Largest |ours - RecBole| per metric and reference."""
    val = val.assign(abs_diff=(val["ours"] - val["recbole"]).abs())
    out = (val.groupby(["metric", "source"], sort=False)["abs_diff"].agg(["size", "max"]).reset_index())
    key, k = out["metric"].str.split("@").str[0], out["metric"].str.split("@").str[1]
    return pd.DataFrame({"Metric": key.map(lambda m: METRIC_INFO[m][0]) + "@" + k, "Compared with": out["source"],
                         "Lists": out["size"], "max $|\\Delta|$": out["max"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=EVAL_SPLITS + ("both",), default="both")
    parser.add_argument("--names", default=None, help="comma-separated list names (default: every exported list)")
    parser.add_argument("--k", type=int, default=TOPK_FINAL)
    parser.add_argument("--min-rating", type=float, default=None,
                        help="count only held-out interactions rated >= this as relevant (default: all)")
    args = parser.parse_args()

    splits = EVAL_SPLITS if args.split == "both" else (args.split,)
    tag = "" if args.min_rating is None else f"_minrating{args.min_rating:g}"
    reported = recbole_reported()
    PER_USER_DIR.mkdir(parents=True, exist_ok=True)
    summaries, validation = [], []
    for split in splits:
        ctx = EvaluationContext.for_split(split, args.min_rating)
        names = args.names.split(",") if args.names else available_names(split)
        if not names:
            raise SystemExit(f"no lists in {RECOMMENDATIONS_DIR} -- run `python -m project.experiments.run_models` first")
        print(f"\n== {split}: {len(ctx.users)} users, {ctx.n_items} items; popularity groups:")
        print(lookups.describe_popularity_groups(ctx.counts, ctx.item_groups).to_string(index=False))
        for name in names:
            recs = load_recommendations(name, split)
            res = evaluate(recs, ctx, args.k, name)
            res.per_user.to_csv(PER_USER_DIR / f"{name}_{split}{tag}_per_user.csv", float_format="%.6g")
            summaries.append(res.summary)
            if tag == "":
                idx, _ = index_matrix(recs, ctx, args.k)
                validation += validation_rows(name, split, res.summary, idx, ctx, args.k, reported)

    summary = pd.DataFrame(summaries)
    for split in splits:
        path = RESULTS_PROCESSED_DIR / f"metrics_{split}{tag}.csv"
        summary[summary["split"] == split].drop(columns="split").to_csv(path, index=False, float_format="%.6g")
        print("wrote", path)
    cols = ["name", "split"] + [f"{m}@{args.k}" for m in ["ndcg", "recall"] + TABLE_METRICS]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if tag == "" and args.names is None and args.split == "both":  # the report assets need the full run
        val = pd.DataFrame(validation)
        val.to_csv(RESULTS_PROCESSED_DIR / "metric_validation.csv", index=False, float_format="%.6g")
        vt = validation_table(val)
        print("\nvalidation against RecBole:\n" + vt.to_string(index=False))
        print("wrote", save_latex_table(vt, "metric_validation", escape=False, float_format="%.1e",
                                        column_format="llrr"))
        rt = results_table(summary, args.k)
        print("wrote", save_latex_table(rt, "beyond_accuracy_results", escape=False, float_format="%.3f",
                                        column_format="l" + "r" * (rt.shape[1] - 1)))


if __name__ == "__main__":
    main()
