"""Build the main results table and figure for the report (Track A, Task 2.2 + appendix).

Reads ``results/processed/recbole_metrics_{quick,tuned}.csv`` (written by run_models) and
``results/processed/tuning_summary.csv`` (written by tune_models) and writes

    report/tables/generated/model_results.tex      test metrics of the tuned models vs. baselines
    report/tables/generated/tuning_summary.tex     search space size, best hyper-parameters, NDCG before/after tuning
    results/processed/model_results.csv            the same table as CSV (one row per model)
    figures/generated/model_comparison.pdf         NDCG@10 / Recall@10 / Precision@10 on test per model

Usage (repository root)::

    python -m project.experiments.results_table            # tuned numbers (falls back to quick for untuned models)
    python -m project.experiments.results_table --mode quick
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from project.models.pipeline import METRIC_COLUMNS
from project.models.registry import ALL_MODELS, MODELS
from project.utils.paths import RESULTS_PROCESSED_DIR
from project.utils.report_assets import save_figure, save_latex_table

PRETTY = {"ndcg@10": "NDCG@10", "recall@10": "Recall@10", "precision@10": "Precision@10",
          "mrr@10": "MRR@10", "hit@10": "Hit@10", "map@10": "MAP@10"}
ORDER = [m for m in ALL_MODELS]  # registry order: baselines first, then by family

# two-colour palette: baselines in neutral grey (hatched), learned models in one blue
COLOR_MODEL, COLOR_BASELINE = "#3B6EA8", "#9A9A9A"


def _load(mode: str) -> pd.DataFrame:
    path = RESULTS_PROCESSED_DIR / f"recbole_metrics_{mode}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def collect(mode: str) -> pd.DataFrame:
    """One row per model: test + valid metrics; untuned models (baselines) come from the quick run."""
    tuned, quick = _load("tuned"), _load("quick")
    src = tuned if mode == "tuned" else quick
    if src.empty:
        raise SystemExit(f"no results for mode={mode}; run `python -m project.experiments.run_models --mode {mode}`")
    frames = [src]
    if mode == "tuned" and not quick.empty:
        frames.append(quick[~quick["model"].isin(src["model"])])  # baselines are not tuned
    df = pd.concat(frames, ignore_index=True)
    df["order"] = df["model"].map({m: i for i, m in enumerate(ORDER)})
    return df.sort_values(["order", "split"]).drop(columns="order")


def results_table(df: pd.DataFrame) -> pd.DataFrame:
    test = df[df.split == "test"].set_index("model")
    valid = df[df.split == "valid"].set_index("model")
    out = pd.DataFrame({"Model": test.index, "Family": [MODELS[m].family for m in test.index]})
    for m in METRIC_COLUMNS:
        out[PRETTY[m]] = test[m].values
    out["valid NDCG@10"] = valid.loc[test.index, "ndcg@10"].values
    return out


def _bold_best(out: pd.DataFrame, cols) -> pd.DataFrame:
    fmt = out.copy()
    for c in cols:
        best = out[c].max()
        fmt[c] = [f"\\textbf{{{v:.4f}}}" if v == best else f"{v:.4f}" for v in out[c]]
    return fmt


def tuning_table(df: pd.DataFrame) -> pd.DataFrame | None:
    path = RESULTS_PROCESSED_DIR / "tuning_summary.csv"
    quick = _load("quick")
    if not path.exists() or quick.empty:
        return None
    summary = pd.read_csv(path).set_index("model")
    rows = []
    for m in ORDER:
        if m not in summary.index:
            continue
        q_valid = quick[(quick.model == m) & (quick.split == "valid")]["ndcg@10"]
        q_test = quick[(quick.model == m) & (quick.split == "test")]["ndcg@10"]
        t_test = df[(df.model == m) & (df.split == "test") & (df["mode"] == "tuned")]["ndcg@10"]
        params = json.loads(summary.loc[m, "best_params"])
        rows.append({
            "Model": m,
            "Configs": int(summary.loc[m, "n_trials"]),
            "Selected hyper-parameters": ", ".join(f"{k}={v}" for k, v in params.items()),
            "valid NDCG@10 untuned": float(q_valid.iloc[0]) if len(q_valid) else float("nan"),
            "valid NDCG@10 tuned": float(summary.loc[m, "valid_ndcg@10_tuned"]),
            "test NDCG@10 untuned": float(q_test.iloc[0]) if len(q_test) else float("nan"),
            "test NDCG@10 tuned": float(t_test.iloc[0]) if len(t_test) else float("nan"),
        })
    return pd.DataFrame(rows)


def comparison_figure(out: pd.DataFrame):
    metrics = ["NDCG@10", "Recall@10", "Precision@10"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(9.5, 3.4), sharey=True)
    models = list(out["Model"])[::-1]  # best-known first at the top when reversed below
    for ax, metric in zip(axes, metrics):
        vals = out.set_index("Model").loc[models, metric]
        colors = [COLOR_BASELINE if MODELS[m].baseline else COLOR_MODEL for m in models]
        bars = ax.barh(models, vals, color=colors, height=0.62, edgecolor="white", linewidth=0.5)
        for b, m in zip(bars, models):
            if MODELS[m].baseline:
                b.set_hatch("////")
        pop = out.set_index("Model").loc["Pop", metric] if "Pop" in out["Model"].values else None
        if pop is not None:
            ax.axvline(pop, color="#555555", linestyle=":", linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=7, color="#333333")
        ax.set_title(metric, fontsize=10)
        ax.set_xlim(0, vals.max() * 1.22)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("")
    fig.suptitle("Test-set accuracy per model (dotted line: Pop baseline; hatched: baselines)", fontsize=9, y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["quick", "tuned"], default="tuned")
    args = parser.parse_args()

    df = collect(args.mode)
    out = results_table(df)
    out.to_csv(RESULTS_PROCESSED_DIR / "model_results.csv", index=False)
    metric_cols = [PRETTY[m] for m in METRIC_COLUMNS] + ["valid NDCG@10"]
    p = save_latex_table(_bold_best(out, metric_cols), "model_results", escape=False, column_format="ll" + "r" * len(metric_cols))
    print("wrote", p)
    p = save_figure(comparison_figure(out), "model_comparison")
    print("wrote", p)
    tt = tuning_table(df)
    if tt is not None:
        tt.to_csv(RESULTS_PROCESSED_DIR / "tuning_table.csv", index=False)
        tt["Selected hyper-parameters"] = tt["Selected hyper-parameters"].str.replace("_", "\\_")
        p = save_latex_table(tt.rename(columns={
            "valid NDCG@10 untuned": "val.\\ untuned", "valid NDCG@10 tuned": "val.\\ tuned",
            "test NDCG@10 untuned": "test untuned", "test NDCG@10 tuned": "test tuned"}),
            "tuning_summary", escape=False, column_format="lrp{5.2cm}rrrr")
        print("wrote", p)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
