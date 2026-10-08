"""Overlap analysis: pairwise Jaccard between models' top-k lists and oracle bounds (Task 2.4).

- Pairwise Jaccard similarity of top-k sets across users
- Oracle: per-user best model (upper bound for switching hybrids)
- Oracle model distribution by user group
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from project.metrics.evaluate import EvaluationContext, evaluate
from project.utils.data_formats import TOPK_FINAL


def _user_topk_sets(recs: pd.DataFrame, k: int) -> Dict[str, set]:
    """Extract top-k item sets per user from a recommendation DataFrame."""
    top = recs[recs["rank"] <= k]
    return {uid: set(grp["item_id"]) for uid, grp in top.groupby("user_id")}


def pairwise_overlap(
    lists: Dict[str, pd.DataFrame],
    k: int = TOPK_FINAL,
) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity of top-k sets across all users.

    Returns a DataFrame with columns: model_a, model_b, mean_jaccard,
    median_jaccard, std_jaccard, mean_overlap_count.
    """
    names = sorted(lists.keys())
    sets_by_model = {name: _user_topk_sets(recs, k) for name, recs in lists.items()}

    all_users = set()
    for s in sets_by_model.values():
        all_users.update(s.keys())

    rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i > j:
                continue
            jaccards = []
            overlaps = []
            for uid in all_users:
                sa = sets_by_model[a].get(uid, set())
                sb = sets_by_model[b].get(uid, set())
                union = sa | sb
                if not union:
                    continue
                inter = sa & sb
                jaccards.append(len(inter) / len(union))
                overlaps.append(len(inter))
            jaccards = np.array(jaccards)
            overlaps = np.array(overlaps)
            rows.append({
                "model_a": a,
                "model_b": b,
                "mean_jaccard": float(jaccards.mean()),
                "median_jaccard": float(np.median(jaccards)),
                "std_jaccard": float(jaccards.std()),
                "mean_overlap_count": float(overlaps.mean()),
            })

    return pd.DataFrame(rows)


def overlap_matrix(
    lists: Dict[str, pd.DataFrame],
    k: int = TOPK_FINAL,
) -> pd.DataFrame:
    """Symmetric (M, M) matrix of mean Jaccard overlap, suitable for heatmap plotting."""
    pw = pairwise_overlap(lists, k)
    names = sorted(lists.keys())
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for _, row in pw.iterrows():
        mat.loc[row["model_a"], row["model_b"]] = row["mean_jaccard"]
        mat.loc[row["model_b"], row["model_a"]] = row["mean_jaccard"]
    return mat


def oracle_analysis(
    lists: Dict[str, pd.DataFrame],
    ctx: EvaluationContext,
    k: int = TOPK_FINAL,
) -> Dict:
    """Compute the oracle (per-user best model) and its performance.

    Returns dict with:
    - 'oracle_recs': the oracle recommendation DataFrame
    - 'oracle_result': EvaluationResult of the oracle
    - 'picks': Series indexed by user_id, values are the best model name
    - 'model_ndcgs': dict of each model's overall NDCG@k
    """
    per_user_ndcg = {}
    per_user_recs = {}
    model_ndcgs = {}

    for name, recs in lists.items():
        result = evaluate(recs, ctx, k=k, name=name)
        model_ndcgs[name] = float(result.summary[f"ndcg@{k}"])
        for uid, ndcg_val in result.per_user[f"ndcg@{k}"].items():
            per_user_ndcg.setdefault(uid, {})[name] = ndcg_val
            per_user_recs.setdefault(uid, {})[name] = recs[
                (recs["user_id"] == uid) & (recs["rank"] <= k)
            ]

    picks = {}
    oracle_parts = []
    for uid, model_scores in per_user_ndcg.items():
        best_model = max(model_scores, key=model_scores.get)
        picks[uid] = best_model
        oracle_parts.append(per_user_recs[uid][best_model])

    picks_series = pd.Series(picks, name="best_model")
    picks_series.index.name = "user_id"

    oracle_recs = pd.concat(oracle_parts, ignore_index=True)
    oracle_result = evaluate(oracle_recs, ctx, k=k, name="Oracle")

    return {
        "oracle_recs": oracle_recs,
        "oracle_result": oracle_result,
        "picks": picks_series,
        "model_ndcgs": model_ndcgs,
    }


def oracle_distribution(
    picks: pd.Series,
    user_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-tabulate oracle model picks with user groups.

    Returns a long DataFrame with columns: dimension, group, model, count, share.
    """
    picks_df = picks.to_frame("model").join(user_groups)
    rows = []
    for dim in user_groups.columns:
        for group in sorted(picks_df[dim].unique()):
            subset = picks_df[picks_df[dim] == group]
            total = len(subset)
            for model, count in subset["model"].value_counts().items():
                rows.append({
                    "dimension": dim,
                    "group": group,
                    "model": model,
                    "count": int(count),
                    "share": count / total if total > 0 else 0.0,
                })
    return pd.DataFrame(rows)
