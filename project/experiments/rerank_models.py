"""Re-rank the best lists and report the trade-off (Tasks 3.1-3.2, Track E).

For every base list (``--models``: a model, or a hybrid once it is exported) and objective
(``--objectives``, see ``project/rerankers/objectives.py``):

1. sweep lam on VALIDATION, top 10 re-ranked, every metric      results/processed/reranker_sweep_valid.csv
2. lam* = knee of NDCG@10 against the objective's target metric  (``project/rerankers/tradeoff.py``)
3. re-rank all 50 candidates of valid and test with lam*         results/raw/recommendations/<Model>+<Objective>_{valid,test}_top50.csv
4. TEST metrics of every base list and its re-ranked lists       results/processed/reranker_results.csv
                                                                 report/tables/generated/reranker_results.tex
5. trade-off curves on validation                                figures/generated/reranker_tradeoff.pdf

It also writes ``results/processed/reranker_pool_test.csv``: the metrics of each base list's full
top 50, i.e. how much diversity, long tail or calibration a re-ranker can find among the candidates.

Usage (repository root)::

    python -m project.experiments.rerank_models
    python -m project.experiments.rerank_models --models SLIMElastic,LightGCN,WeightedHybrid
    python -m project.experiments.rerank_models --objectives MMR,xQuAD --lambdas 0,0.05,0.1,0.2,0.4,0.7,1
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from project.metrics import METRIC_INFO, EvaluationContext, evaluate
from project.rerankers import DEFAULT_LAMBDAS, OBJECTIVES, knee, make_objective, rerank, reranked_name, sweep
from project.utils.data_formats import (EVAL_SPLITS, TOPK_CANDIDATES, TOPK_FINAL, load_recommendations,
                                        recommendations_path, save_recommendations)
from project.utils.paths import RESULTS_PROCESSED_DIR
from project.utils.report_assets import save_figure, save_latex_table

# The best linear item-item model and the best embedding model on validation NDCG@10
# (results/processed/metrics_valid.csv); they differ most in popularity bias (tail share 0.002 vs 0.021).
DEFAULT_MODELS = ["SLIMElastic", "LightGCN"]
TABLE_METRICS = ["ndcg", "ild", "miscalibration", "upd", "tailshare", "coverage", "gini"]
SHORT_LABELS = {"miscalibration": "MC", "tailshare": "Tail", "coverage": "Cov."}


def check_inputs(models: List[str]):
    missing = [str(recommendations_path(m, s)) for m in models for s in EVAL_SPLITS
               if not recommendations_path(m, s).exists()]
    if missing:
        raise SystemExit("missing candidate lists (run Track A's pipeline or unzip the released results/raw):\n  "
                         + "\n  ".join(missing))


def run_sweeps(models, objectives, lambdas, ctx: EvaluationContext) -> pd.DataFrame:
    curves = []
    for model in models:
        recs = load_recommendations(model, "valid")
        for objective in objectives:
            curve = sweep(recs, objective, ctx, lambdas, TOPK_FINAL, name=model)
            curves.append(curve)
            target = f"{objective.target_metric}@{TOPK_FINAL}"
            print(f"{model:>12s} {objective.name:<13s} " + "  ".join(
                f"{lam:g}:{n:.3f}/{t:.3f}" for lam, n, t in zip(curve["lambda"], curve[f"ndcg@{TOPK_FINAL}"], curve[target])))
    return pd.concat(curves, ignore_index=True)


def choose_lambdas(curves: pd.DataFrame) -> Dict[tuple, float]:
    return {(model, obj): knee(curve, OBJECTIVES[obj].target_metric)
            for (model, obj), curve in curves.groupby(["name", "objective"], sort=False)}


def results_table(results: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Model": results["model"].str.replace("_", r"\_"),
                        "Re-ranker": results["objective"].fillna("--"),
                        r"$\lambda$": results["lambda"].map(lambda v: "--" if pd.isna(v) else f"{v:g}")})
    for m in TABLE_METRICS:
        arrow = r"$\uparrow$" if METRIC_INFO[m][1] else r"$\downarrow$"
        out[f"{SHORT_LABELS.get(m, METRIC_INFO[m][0])} {arrow}"] = results[f"{m}@{TOPK_FINAL}"].values
    return out


def tradeoff_figure(curves: pd.DataFrame, chosen: Dict[tuple, float], objectives):
    """One panel per objective: NDCG@10 against the target metric over lam, the chosen lam starred.
    Axes of lower-is-better metrics are inverted, so 'better' is always to the right."""
    fig, axes = plt.subplots(1, len(objectives), figsize=(7.0, 2.0), squeeze=False, sharey=True)
    for ax, objective in zip(axes[0], objectives):
        target = f"{objective.target_metric}@{TOPK_FINAL}"
        for model, curve in curves[curves["objective"] == objective.name].groupby("name", sort=False):
            line, = ax.plot(curve[target], curve[f"ndcg@{TOPK_FINAL}"], marker="o", markersize=2.5, linewidth=1,
                            label=model)
            at = curve[curve["lambda"] == chosen[(model, objective.name)]]
            ax.plot(at[target], at[f"ndcg@{TOPK_FINAL}"], marker="*", markersize=9, color=line.get_color(),
                    linestyle="none")
        label, higher = METRIC_INFO[objective.target_metric]
        ax.set_title(objective.name, fontsize=8)
        ax.set_xlabel(f"{label}@{TOPK_FINAL} " + ("(higher better)" if higher else "(lower better)"), fontsize=7)
        if not higher:
            ax.invert_xaxis()
        ax.tick_params(labelsize=6)
    axes[0][0].set_ylabel(f"NDCG@{TOPK_FINAL}", fontsize=7)
    axes[0][0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated base list names")
    parser.add_argument("--objectives", default=",".join(OBJECTIVES), help="comma-separated, from " + ",".join(OBJECTIVES))
    parser.add_argument("--lambdas", default=",".join(f"{v:g}" for v in DEFAULT_LAMBDAS))
    args = parser.parse_args()

    models = args.models.split(",")
    objectives = [make_objective(name) for name in args.objectives.split(",")]
    lambdas = [float(v) for v in args.lambdas.split(",")]
    check_inputs(models)
    ctx = {split: EvaluationContext.for_split(split) for split in EVAL_SPLITS}

    print(f"== validation sweep (lam: NDCG@{TOPK_FINAL} / target metric)")
    curves = run_sweeps(models, objectives, lambdas, ctx["valid"])
    curves.to_csv(RESULTS_PROCESSED_DIR / "reranker_sweep_valid.csv", index=False, float_format="%.6g")
    chosen = choose_lambdas(curves)

    print("\n== re-ranking all candidates with the chosen lam")
    rows, pool = [], []
    for model in models:
        base_test = load_recommendations(model, "test")
        rows.append({"model": model, "objective": None, "lambda": None,
                     **evaluate(base_test, ctx["test"], TOPK_FINAL, model).summary.drop(["name", "split", "k"]).to_dict()})
        pool.append(evaluate(base_test, ctx["test"], TOPK_CANDIDATES, model).summary)
        for objective in objectives:
            lam = chosen[(model, objective.name)]
            name = reranked_name(model, objective)
            for split in EVAL_SPLITS:
                recs = rerank(load_recommendations(model, split), objective, lam, ctx[split])
                print("wrote", save_recommendations(recs, name, split), f"(lam = {lam:g})")
                if split == "test":
                    rows.append({"model": model, "objective": objective.name, "lambda": lam,
                                 **evaluate(recs, ctx["test"], TOPK_FINAL, name).summary.drop(["name", "split", "k"]).to_dict()})

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PROCESSED_DIR / "reranker_results.csv", index=False, float_format="%.6g")
    pd.DataFrame(pool).to_csv(RESULTS_PROCESSED_DIR / "reranker_pool_test.csv", index=False, float_format="%.6g")
    table = results_table(results)
    print("\n== test\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("wrote", save_latex_table(table, "reranker_results", escape=False, float_format="%.3f",
                                    column_format="llr" + "r" * len(TABLE_METRICS)))
    print("wrote", save_figure(tradeoff_figure(curves, chosen, objectives), "reranker_tradeoff"))
    plt.close("all")


if __name__ == "__main__":
    main()
