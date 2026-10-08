"""Build, tune, export and analyse the weighted hybrids (Track C: Tasks 1.3, 1.5, 2.3, 2.6).

Reads the members' released score matrices and lists (``results/raw``, Track A) and writes

    results/raw/scores/<Hybrid>_{valid,test}.npz, results/raw/recommendations/<Hybrid>_{valid,test}_top50.csv
        for WeightedHybrid (the plan's shortlist of members), WeightedHybridAll (every model) and
        GroupHybrid (Task 2.6) -- Track B's evaluate_models / group_analysis pick them up by name
    results/processed/hybrid_tuning.csv          every setting: cross-validated NDCG@10 on validation (Task 1.5)
    results/processed/hybrid_weights.csv         learned weights with bootstrap intervals (Tasks 1.3, 2.3)
    results/processed/hybrid_ablation.csv        leave-one-member-out, single members, all models, + Pop (Task 2.3)
    results/processed/hybrid_attribution.csv     hit attribution per member; _counts.csv: hits by number of members
    results/processed/hybrid_member_overlap.csv  score correlation and top-10 overlap between members
    results/processed/hybrid_groups.csv          group-aware vs. global hybrid per user group (Task 2.6)
    results/processed/hybrid_group_weights.csv   the per-group weights
    results/processed/hybrid_results.csv         summary: best member vs. the hybrids, validation and test
    results/processed/hybrid_significance.csv    paired per-user NDCG@10 differences: hybrid vs. best member, bootstrap + Wilcoxon
    figures/generated/hybrid_coefficients.pdf, report/tables/generated/{hybrid_results,hybrid_ablation,hybrid_groups}.tex

Usage (repository root)::

    python -m project.experiments.hybrid                             # full run
    python -m project.experiments.hybrid --grid quick                # small grid (smoke test)
    python -m project.experiments.hybrid --setting zscore,ridge,0.001,union    # skip tuning
    python -m project.experiments.hybrid --members ItemKNN,EASE,BPR,NeuMF,LightGCN,ContentBased
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from project.analysis.coefficients import bootstrap_weights, hit_attribution, member_overlap
from project.analysis.group_analysis import bootstrap_ci
from project.hybrids.weighted_tuning import HybridData, Setting, cv_score, default_grid, fit_hybrid, tune
from project.hybrids.weighted import (DEFAULT_MEMBERS, aligned_scores, combine_by_group, cross_fitted, fit_group_weights,
                                      lists_from, member_top_lists, score_matrix, user_folds)
from project.metrics import EvaluationContext, evaluate, lookups
from project.models.registry import ALL_MODELS, MODELS
from project.utils.data_formats import (EVAL_SPLITS, TOPK_CANDIDATES, TOPK_FINAL, load_recommendations, save_recommendations,
                                        save_scores, scores_path)
from project.utils.paths import RESULTS_PROCESSED_DIR
from project.utils.report_assets import save_figure, save_latex_table

K = TOPK_FINAL
NAMES = {"main": "WeightedHybrid", "all": "WeightedHybridAll", "group": "GroupHybrid"}
DIMENSIONS = {"activity": lookups.ACTIVITY_GROUPS, "taste": lookups.TASTE_GROUPS}
SUMMARY_METRICS = ["ndcg", "recall", "precision", "mrr", "map", "coverage", "avgpop", "tailshare", "upd",
                   "miscalibration", "gini", "novelty", "ild"]

# figure palette, shared with project/experiments/group_analysis.py (dataviz reference palette)
BLUE, RAMP = "#2a78d6", ["#86b6ef", "#2a78d6", "#104281"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"


# ------------------------------------------------------------------------------------------ inputs
def available(name: str) -> bool:
    return all(scores_path(name, s).exists() for s in EVAL_SPLITS)


def resolve_members(arg: str, all_models: List[str]) -> List[str]:
    names = DEFAULT_MEMBERS if arg == "default" else all_models if arg == "all" else [n.strip() for n in arg.split(",") if n.strip()]
    missing = [n for n in names if not available(n)]
    if missing:
        raise SystemExit(f"no score files for {missing} in results/raw/scores -- Track A's release is needed "
                         "(project/models/README.md)")
    return names


def load_inputs(names: List[str]) -> HybridData:
    ctx_v, ctx_t = EvaluationContext.for_split("valid"), EvaluationContext.for_split("test")
    raw_v = {n: aligned_scores(n, "valid", ctx_v) for n in names}
    raw_t = {n: aligned_scores(n, "test", ctx_t) for n in names}
    lists_v = {n: member_top_lists(n, "valid", ctx_v, TOPK_CANDIDATES, raw_v[n]) for n in names}
    return HybridData(ctx_v, raw_v, lists_v, ctx_t, raw_t)


def score_lists(hyb: np.ndarray, ctx: EvaluationContext, name: str):
    return evaluate(lists_from(hyb, ctx, K), ctx, K, name)


def export(name: str, hyb_valid: np.ndarray, hyb_test: np.ndarray, data: HybridData) -> None:
    for split, hyb in (("valid", hyb_valid), ("test", hyb_test)):
        ctx = data.ctx[split]
        save_scores(score_matrix(hyb, ctx), name, split)
        save_recommendations(lists_from(hyb, ctx, TOPK_CANDIDATES), name, split, TOPK_CANDIDATES)
    print(f"exported {name} (scores + top-{TOPK_CANDIDATES} lists, valid and test)")


def user_labels(ctx: EvaluationContext, dim: str) -> np.ndarray:
    return ctx.user_groups.reindex(ctx.users)[dim].to_numpy()


# --------------------------------------------------------------------------------------- analyses
def ablation(data: HybridData, members: List[str], setting: Setting, folds: np.ndarray, all_models: List[str],
             grid: List[Setting]) -> pd.DataFrame:
    """NDCG@10 (cross-validated on validation, and on test) of the full hybrid, without each member,
    of each member alone, of all models (tuned on its own grid) and of the shortlist + Pop."""
    ctx_v, ctx_t = data.ctx["valid"], data.ctx["test"]
    variants = [("full", members)] + [(f"without {m}", [x for x in members if x != m]) for m in members]
    variants += [(f"only {m}", [m]) for m in members]
    if sorted(all_models) != sorted(members):
        variants.append(("all models", all_models))
    if "Pop" in data.raw["valid"] and "Pop" not in members:
        variants.append(("+ Pop", members + ["Pop"]))
    rows = []
    for label, mem in variants:
        t0 = time.time()
        used = setting
        if len(mem) == 1:  # a member alone: its own exported lists (nothing to fit)
            cv = evaluate(load_recommendations(mem[0], "valid"), ctx_v, K, mem[0]).summary
            test = evaluate(load_recommendations(mem[0], "test"), ctx_t, K, mem[0]).summary
            w = None
        else:
            if label == "all models":  # its own tuning, so the comparison is fair to the larger member set
                t_all, used = tune(data, mem, grid, folds, verbose=False)
                cv = pd.Series({f"ndcg@{K}": t_all.iloc[0][f"cv_ndcg@{K}"]})
            else:
                cv = cv_score(data, mem, setting, folds)
            w, hyb_v, hyb_t = fit_hybrid(data, mem, used)
            test = score_lists(hyb_t, ctx_t, label).summary
            if label == "all models":
                export(NAMES["all"], hyb_v, hyb_t, data)
        rows.append({"variant": label, "members": ",".join(mem), "n_members": len(mem), "setting": used.label,
                     f"cv_ndcg@{K}": cv[f"ndcg@{K}"], f"test_ndcg@{K}": test[f"ndcg@{K}"],
                     f"test_recall@{K}": test[f"recall@{K}"], f"test_coverage@{K}": test[f"coverage@{K}"],
                     f"test_avgpop@{K}": test[f"avgpop@{K}"],
                     "weights_std": "" if w is None else ";".join(f"{m}={v:.4f}" for m, v in zip(w.members, w.coef_std))})
        print(f"  ablation {label:22s} CV {cv[f'ndcg@{K}']:.4f}  test {test[f'ndcg@{K}']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    full = out.iloc[0]
    out["delta_test"] = out[f"test_ndcg@{K}"] - full[f"test_ndcg@{K}"]
    out["delta_cv"] = out[f"cv_ndcg@{K}"] - full[f"cv_ndcg@{K}"]
    return out


def group_aware(data: HybridData, members: List[str], setting: Setting, folds: np.ndarray, global_test):
    """Per-group weights for every dimension of Task 2.5; the dimension with the best cross-validated
    validation NDCG@10 is exported as GroupHybrid. Returns (groups table, weights table, chosen dimension)."""
    ctx_v, ctx_t = data.ctx["valid"], data.ctx["test"]
    table = data.table(members, setting)
    cv_global = cv_score(data, members, setting, folds)[f"ndcg@{K}"]
    per_dim, weight_rows, group_rows = {}, [], []
    for dim, groups in DIMENSIONS.items():
        lab_v, lab_t = user_labels(ctx_v, dim), user_labels(ctx_t, dim)
        cv = cv_score(data, members, setting, folds, user_labels=lab_v, labels=groups)[f"ndcg@{K}"]
        wg = fit_group_weights(table, members, lab_v, groups, setting.regression, setting.lam,
                               rescale=setting.rescale, candidates=setting.candidates)
        hyb_v = combine_by_group(data.scores(members, "valid", setting.rescale), wg, lab_v)
        hyb_t = combine_by_group(data.scores(members, "test", setting.rescale), wg, lab_t)
        res = score_lists(hyb_t, ctx_t, f"GroupHybrid-{dim}")
        per_dim[dim] = (cv, hyb_v, hyb_t, res)
        print(f"  group-aware by {dim}: CV NDCG@{K} {cv:.4f} (global {cv_global:.4f}); test {res.summary[f'ndcg@{K}']:.4f}", flush=True)
        for g, w in wg.items():
            for m, c, s in zip(members, w.coef_std, w.share):
                weight_rows.append({"dimension": dim, "group": g, "users": int((lab_v == g).sum()), "member": m,
                                    "coef_std": c, "share": s})
        pu_g, pu_w = res.per_user, global_test.per_user.reindex(res.per_user.index)
        for g in ("all",) + tuple(groups):
            mask = np.ones(len(pu_g), dtype=bool) if g == "all" else (pu_g[dim] == g).to_numpy()
            d_ndcg = (pu_g[f"ndcg@{K}"] - pu_w[f"ndcg@{K}"]).to_numpy()[mask]
            lo, hi = bootstrap_ci(d_ndcg)
            group_rows.append({"dimension": dim, "group": g, "users": int(mask.sum()),
                               f"cv_ndcg@{K}_global": cv_global, f"cv_ndcg@{K}_group": cv,
                               f"ndcg@{K}_global": pu_w[f"ndcg@{K}"][mask].mean(), f"ndcg@{K}_group": pu_g[f"ndcg@{K}"][mask].mean(),
                               "diff_ndcg": d_ndcg.mean(), "diff_ci_low": lo, "diff_ci_high": hi,
                               f"recall@{K}_global": pu_w[f"recall@{K}"][mask].mean(), f"recall@{K}_group": pu_g[f"recall@{K}"][mask].mean(),
                               f"upd@{K}_global": pu_w[f"upd@{K}"][mask].mean(), f"upd@{K}_group": pu_g[f"upd@{K}"][mask].mean(),
                               f"avgpop@{K}_global": pu_w[f"avgpop@{K}"][mask].mean(), f"avgpop@{K}_group": pu_g[f"avgpop@{K}"][mask].mean()})
    best_dim = max(per_dim, key=lambda d: per_dim[d][0])
    export(NAMES["group"], per_dim[best_dim][1], per_dim[best_dim][2], data)
    groups_df = pd.DataFrame(group_rows)
    groups_df["exported"] = groups_df["dimension"] == best_dim
    return groups_df, pd.DataFrame(weight_rows), best_dim


def paired_test(a: pd.Series, b: pd.Series, comparison: str, split: str) -> dict:
    """Mean per-user difference a - b with the 95 % percentile bootstrap of Task 2.5 and a Wilcoxon signed-rank test."""
    d = (a - b.reindex(a.index)).to_numpy()
    lo, hi = bootstrap_ci(d)
    nonzero = d[d != 0]
    return {"comparison": comparison, "split": split, "metric": f"ndcg@{K}", "users": len(d), "mean_diff": d.mean(),
            "ci_low": lo, "ci_high": hi, "users_better": int((d > 0).sum()), "users_worse": int((d < 0).sum()),
            "wilcoxon_p": float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else np.nan}


def significance(data: HybridData, members: List[str], setting: Setting, folds: np.ndarray, best_member: str) -> pd.DataFrame:
    """Is a hybrid's gain real? Validation uses the cross-fitted lists (weights never saw the user), test the exported ones."""
    ctx_v = data.ctx["valid"]
    per_user = lambda name, split: evaluate(load_recommendations(name, split), data.ctx[split], K, name).per_user[f"ndcg@{K}"]
    hyb_cv = cross_fitted(data.scores(members, "valid", setting.rescale), data.table(members, setting), folds,
                          setting.regression, setting.lam)
    cv = evaluate(lists_from(hyb_cv, ctx_v, K), ctx_v, K).per_user[f"ndcg@{K}"]
    rows = [paired_test(cv, per_user(best_member, "valid"), f"{NAMES['main']} - {best_member}", "valid (cross-fitted)"),
            paired_test(per_user(NAMES["main"], "test"), per_user(best_member, "test"), f"{NAMES['main']} - {best_member}", "test")]
    if available(NAMES["all"]):
        rows.append(paired_test(per_user(NAMES["all"], "test"), per_user(best_member, "test"), f"{NAMES['all']} - {best_member}", "test"))
    rows.append(paired_test(per_user(NAMES["group"], "test"), per_user(NAMES["main"], "test"), f"{NAMES['group']} - {NAMES['main']}", "test"))
    return pd.DataFrame(rows)


def summary_table(data: HybridData, best_member: str, cv_values: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for name in [best_member, NAMES["main"], NAMES["all"], NAMES["group"]]:
        if not available(name):
            continue
        row = {"name": name, f"cv_valid_ndcg@{K}": cv_values.get(name, np.nan)}
        for split in EVAL_SPLITS:
            s = evaluate(load_recommendations(name, split), data.ctx[split], K, name).summary
            for m in (SUMMARY_METRICS if split == "test" else ["ndcg", "recall"]):
                row[f"{split}_{m}@{K}"] = s[f"{m}@{K}"]
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------- assets
def _style(ax, title):
    ax.set_title(title, fontsize=7.5, color=INK, loc="left", pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", labelsize=7, colors=INK_2, length=0)
    ax.tick_params(axis="x", colors=MUTED)


def coefficients_figure(weights: pd.DataFrame, ablation: pd.DataFrame, group_w: pd.DataFrame, dim: str, members: List[str]):
    y = np.arange(len(members))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 0.3 * len(members) + 1.25), sharey=True,
                             gridspec_kw={"width_ratios": [1.05, 1.25, 1.05]})
    w = weights.set_index("member").loc[members]
    ax = axes[0]
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.hlines(y, w["share_ci_low"], w["share_ci_high"], color=BLUE, linewidth=1.2)
    ax.scatter(w["share"], y, s=30, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.set_yticks(y, members)
    _style(ax, "(a) weight share (95% bootstrap CI)")
    ax = axes[1]
    d = ablation.set_index("variant")
    vals = np.array([d.loc[f"without {m}", "delta_test"] for m in members])
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.barh(y, vals, color=BLUE, height=0.55, edgecolor=SURFACE, linewidth=0.5)
    span = max(abs(vals).max(), 1e-3)
    for yy, v in zip(y, vals):
        ax.text(v - 0.04 * span if v < 0 else v + 0.04 * span, yy, f"{v:+.4f}", va="center",
                ha="right" if v < 0 else "left", fontsize=6.5, color=INK_2)
    ax.set_xlim(-span * 1.9, span * 1.9)
    _style(ax, f"(b) $\\Delta$ test NDCG@{K}, member removed")
    ax = axes[2]
    groups = DIMENSIONS[dim]
    offsets = np.linspace(0.22, -0.22, len(groups))
    ax.axvline(0, color=AXIS, linewidth=0.8)
    for j, g in enumerate(groups):
        gw = group_w[(group_w["dimension"] == dim) & (group_w["group"] == g)].set_index("member").loc[members, "share"]
        ax.scatter(gw.values, y + offsets[j], s=22, color=RAMP[j], edgecolor=SURFACE, linewidth=0.7, zorder=3, label=g)
    _style(ax, f"(c) share per {dim} group")
    ax.legend(loc="lower right", frameon=False, fontsize=6, handletextpad=0.1, labelcolor=INK_2, markerscale=0.9)
    fig.tight_layout(w_pad=1.4)
    return fig


def results_tex(summary: pd.DataFrame) -> pd.DataFrame:
    cols = {"name": "Model", f"cv_valid_ndcg@{K}": "val.\\ CV", f"valid_ndcg@{K}": "val.",
            f"test_ndcg@{K}": "NDCG", f"test_recall@{K}": "Recall", f"test_precision@{K}": "Prec.",
            f"test_mrr@{K}": "MRR", f"test_coverage@{K}": "Cov.", f"test_avgpop@{K}": "AvgPop",
            f"test_upd@{K}": "UPD", f"test_miscalibration@{K}": "MC$_{KL}$"}
    out = summary[list(cols)].rename(columns=cols)
    out["Model"] = out["Model"].str.replace("_", r"\_")
    return out


def ablation_tex(weights: pd.DataFrame, ablation: pd.DataFrame, attribution: pd.DataFrame, members: List[str]) -> pd.DataFrame:
    w, d, a = weights.set_index("member"), ablation.set_index("variant"), attribution.set_index("member")
    rows = []
    for m in members:
        rows.append({"Member": m, "weight share": f"{w.loc[m, 'share']:.2f}",
                     "95\\% CI": f"[{w.loc[m, 'share_ci_low']:.2f}, {w.loc[m, 'share_ci_high']:.2f}]",
                     "$\\Delta$NDCG test": f"{d.loc[f'without {m}', 'delta_test']:+.4f}",
                     "$\\Delta$NDCG CV": f"{d.loc[f'without {m}', 'delta_cv']:+.4f}",
                     "contrib.": f"{a.loc[m, 'mean_contribution_share']:.0%}".replace("%", "\\%"),
                     "alone, test": f"{d.loc[f'only {m}', f'test_ndcg@{K}']:.4f}"})
    return pd.DataFrame(rows)


def tuning_tex(tuning: pd.DataFrame) -> pd.DataFrame:
    """Best cross-validated NDCG@10 over the regularisation values, per rescaling x (candidate rule, regression)."""
    best = tuning.groupby(["rescale", "candidates", "regression"], sort=False)[f"cv_ndcg@{K}"].max()
    out = pd.DataFrame({"rescaling": ["zscore", "minmax", "rank"]})
    for cand in ("union", "all"):
        for reg in ("ridge", "nnls", "logistic"):
            out[f"{reg} ({cand})"] = [best.get((r, cand, reg), np.nan) for r in out["rescaling"]]
    return out


def overlap_tex(overlap: pd.DataFrame, members: List[str]) -> pd.DataFrame:
    """Member x member: score correlation above the diagonal, top-10 Jaccard overlap below it."""
    cells = {m: {n: "--" for n in members} for m in members}
    for _, r in overlap.iterrows():
        cells[r["member_a"]][r["member_b"]] = f"{r['score_correlation']:.2f}"
        cells[r["member_b"]][r["member_a"]] = f"{r[f'top{K}_jaccard']:.2f}"
    return pd.DataFrame([{"": m, **cells[m]} for m in members])


def groups_tex(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in groups.iterrows():
        rows.append({"Dimension": r["dimension"], "Group": r["group"], "Users": int(r["users"]),
                     "NDCG global": f"{r[f'ndcg@{K}_global']:.4f}", "NDCG group-aware": f"{r[f'ndcg@{K}_group']:.4f}",
                     "$\\Delta$ [95\\% CI]": f"{r['diff_ndcg']:+.4f} [{r['diff_ci_low']:+.4f}, {r['diff_ci_high']:+.4f}]",
                     "UPD global": f"{r[f'upd@{K}_global']:.3f}", "UPD group-aware": f"{r[f'upd@{K}_group']:.3f}"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", default="default", help='"default" (the plan\'s shortlist), "all" or comma-separated names')
    parser.add_argument("--grid", choices=["full", "quick"], default="full")
    parser.add_argument("--setting", default=None, help='skip tuning: "rescale,regression,lam,candidates"')
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--no-assets", action="store_true", help="skip the figure and LaTeX tables")
    parser.add_argument("--assets-only", action="store_true",
                        help="only rebuild the figure and LaTeX tables from results/processed/hybrid_*.csv")
    args = parser.parse_args()
    t_start = time.time()
    if args.assets_only:
        write_assets(*[pd.read_csv(RESULTS_PROCESSED_DIR / f"hybrid_{n}.csv") for n in
                       ("weights", "ablation", "group_weights", "groups", "results", "attribution", "tuning",
                        "member_overlap")])
        return

    all_models = [m for m in ALL_MODELS if not MODELS[m].baseline and available(m)]
    members = resolve_members(args.members, all_models)
    all_models += [m for m in members if m not in all_models]
    names = list(dict.fromkeys(members + all_models + (["Pop"] if available("Pop") else [])))
    print(f"members: {members}\nall models (ablation): {all_models}")
    data = load_inputs(names)
    ctx_v, ctx_t = data.ctx["valid"], data.ctx["test"]
    folds = user_folds(len(ctx_v.users))
    out_dir = RESULTS_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. tuning (Task 1.5): cross-fitted over users on the validation split
    if args.setting:
        best = Setting.parse(args.setting)
        tuning, _ = tune(data, members, [best], folds)
    else:
        print(f"tuning {len(default_grid(args.grid == 'quick'))} settings on validation ({len(folds)} users, 5 folds)")
        tuning, best = tune(data, members, default_grid(args.grid == "quick"), folds)
    tuning.to_csv(out_dir / "hybrid_tuning.csv", index=False, float_format="%.6g")
    print(f"selected: {best.label}  (CV NDCG@{K} {tuning.iloc[0][f'cv_ndcg@{K}']:.4f})")

    # 2. the weighted hybrid (Task 1.3): weights on all validation users, applied to test
    w, hyb_v, hyb_t = fit_hybrid(data, members, best)
    export(NAMES["main"], hyb_v, hyb_t, data)
    res_valid, res_test = score_lists(hyb_v, ctx_v, NAMES["main"]), score_lists(hyb_t, ctx_t, NAMES["main"])
    print(f"{NAMES['main']}: valid NDCG@{K} {res_valid.summary[f'ndcg@{K}']:.4f} (in-sample) | test {res_test.summary[f'ndcg@{K}']:.4f}")
    n_boot = args.n_boot if not (best.regression == "logistic" and best.candidates == "all") else min(args.n_boot, 50)
    boot = bootstrap_weights(data.table(members, best), members, best.regression, best.lam, n_boot)
    weights = w.as_frame().merge(boot, on="member")
    total = weights["coef_std"].sum()  # share = coef_std / total; the interval is rescaled the same way
    weights["share_ci_low"], weights["share_ci_high"] = weights["ci_low"] / total, weights["ci_high"] / total
    for key, val in (("rescale", best.rescale), ("regression", best.regression), ("lam", best.lam),
                     ("candidates", best.candidates), ("rows", w.n_rows), ("positives", w.n_positive), ("n_boot", n_boot)):
        weights[key] = val
    weights.to_csv(out_dir / "hybrid_weights.csv", index=False, float_format="%.6g")
    print(weights[["member", "coef_std", "share", "share_ci_low", "share_ci_high"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 3. ablations (Task 2.3): leave-one-out, single members, all models, + Pop
    abl = ablation(data, members, best, folds, all_models, default_grid(args.grid == "quick"))
    abl.to_csv(out_dir / "hybrid_ablation.csv", index=False, float_format="%.6g")

    # 4. hit attribution and member overlap (Task 2.3)
    lists_t = [member_top_lists(m, "test", ctx_t, TOPK_CANDIDATES, data.raw["test"][m]) for m in members]
    attribution, counts = hit_attribution(data.scores(members, "test", best.rescale), w, hyb_t, ctx_t, lists_t)
    attribution.to_csv(out_dir / "hybrid_attribution.csv", index=False, float_format="%.6g")
    counts.to_csv(out_dir / "hybrid_attribution_counts.csv", index=False, float_format="%.6g")
    overlap = member_overlap(data.scores(members, "valid", best.rescale), data.candidates(members, "union"),
                             [data.lists_valid[m] for m in members])
    overlap.to_csv(out_dir / "hybrid_member_overlap.csv", index=False, float_format="%.6g")
    print(attribution.to_string(index=False))
    print(counts.to_string(index=False))

    # 5. group-aware hybrid (Task 2.6)
    groups, group_w, best_dim = group_aware(data, members, best, folds, res_test)
    groups.to_csv(out_dir / "hybrid_groups.csv", index=False, float_format="%.6g")
    group_w.to_csv(out_dir / "hybrid_group_weights.csv", index=False, float_format="%.6g")

    # 6. summary (hybrids vs. the best member on validation)
    member_valid = {m: evaluate(load_recommendations(m, "valid"), ctx_v, K, m).summary[f"ndcg@{K}"] for m in members}
    best_member = max(member_valid, key=member_valid.get)
    cv_values = {NAMES["main"]: tuning.iloc[0][f"cv_ndcg@{K}"],
                 NAMES["all"]: abl.set_index("variant")[f"cv_ndcg@{K}"].get("all models", np.nan),
                 NAMES["group"]: groups[groups["exported"]][f"cv_ndcg@{K}_group"].iloc[0],
                 best_member: member_valid[best_member]}
    summary = summary_table(data, best_member, cv_values)
    summary.to_csv(out_dir / "hybrid_results.csv", index=False, float_format="%.6g")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sig = significance(data, members, best, folds, best_member)
    sig.to_csv(out_dir / "hybrid_significance.csv", index=False, float_format="%.6g")
    print(sig.round(4).to_string(index=False))

    # 7. report assets
    if not args.no_assets:
        write_assets(weights, abl, group_w, groups, summary, attribution, tuning, overlap)
    print(f"done in {time.time() - t_start:.0f}s")


def write_assets(weights, abl, group_w, groups, summary, attribution, tuning, overlap):
    members = list(weights["member"])
    best_dim = groups.loc[groups["exported"], "dimension"].iloc[0]
    print("wrote", save_latex_table(tuning_tex(tuning), "hybrid_tuning", escape=False, float_format="%.4f",
                                    column_format="l" + "r" * 6))
    print("wrote", save_latex_table(overlap_tex(overlap, members), "hybrid_overlap", escape=False, column_format="l" + "r" * len(members)))
    print("wrote", save_figure(coefficients_figure(weights, abl, group_w, best_dim, members), "hybrid_coefficients"))
    print("wrote", save_latex_table(results_tex(summary), "hybrid_results", escape=False, float_format="%.4f",
                                    column_format="l" + "r" * 10))
    print("wrote", save_latex_table(ablation_tex(weights, abl, attribution, members), "hybrid_ablation", escape=False,
                                    column_format="lrlrrrr"))
    print("wrote", save_latex_table(groups_tex(groups[groups["exported"]]), "hybrid_groups", escape=False,
                                    column_format="llrrrlrr"))
    plt.close("all")


if __name__ == "__main__":
    main()
