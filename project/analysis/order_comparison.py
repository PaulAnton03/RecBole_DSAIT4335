"""Order comparison: re-rank-then-combine vs combine-then-re-rank (Task 3.3).

Order A: apply a reranker to each individual model's list, then combine the
         re-ranked lists using positional aggregation (Borda count).
Order B: first combine raw model lists (e.g. via RRF), then apply the reranker
         to the combined candidate list.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from project.metrics.evaluate import EvaluationContext, evaluate
from project.utils.data_formats import (
    RECOMMENDATION_COLUMNS,
    TOPK_CANDIDATES,
    TOPK_FINAL,
    load_recommendations,
)


def rerank_then_combine(
    models: List[str],
    split: str,
    reranker_fn: Callable,
    k_rerank: int = TOPK_FINAL,
    k_combine: int = TOPK_FINAL,
    k_candidates: int = TOPK_CANDIDATES,
) -> pd.DataFrame:
    """Order A: rerank each model individually, then combine by Borda count.

    Parameters
    ----------
    models : list of model names
    split : "valid" or "test"
    reranker_fn : callable(recs_df, lam, k_out, k_in) -> recs_df
        Reranker function (e.g. mmr_rerank).
    k_rerank : items to keep after reranking each model
    k_combine : final list length after combining
    k_candidates : candidate list length to load
    """
    per_user_scores: Dict[str, Dict[str, float]] = {}

    for model in models:
        recs = load_recommendations(model, split, k_candidates)
        reranked = reranker_fn(recs, k_out=k_rerank, k_in=k_candidates)

        for _, row in reranked.iterrows():
            uid = row["user_id"]
            iid = row["item_id"]
            borda = k_rerank - row["rank"] + 1
            per_user_scores.setdefault(uid, {}).setdefault(iid, 0.0)
            per_user_scores[uid][iid] += borda

    rows = []
    for uid, item_scores in per_user_scores.items():
        sorted_items = sorted(item_scores.items(), key=lambda x: -x[1])[:k_combine]
        for rank, (iid, score) in enumerate(sorted_items, start=1):
            rows.append((uid, iid, rank, score))

    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def combine_then_rerank(
    models: List[str],
    split: str,
    combine_fn: Callable,
    reranker_fn: Callable,
    k_combine: int = TOPK_CANDIDATES,
    k_final: int = TOPK_FINAL,
) -> pd.DataFrame:
    """Order B: combine raw lists first, then rerank the combined list.

    Parameters
    ----------
    models : list of model names
    split : "valid" or "test"
    combine_fn : callable(models, split, k) -> recs_df
        Combiner function (e.g. mixed_rrf).
    reranker_fn : callable(recs_df, k_out, k_in) -> recs_df
        Reranker function (e.g. mmr_rerank).
    k_combine : candidate list length from the combiner
    k_final : final list length after reranking
    """
    combined = combine_fn(models, split, k=k_combine)
    reranked = reranker_fn(combined, k_out=k_final, k_in=k_combine)
    return reranked


def compare_orders(
    models: List[str],
    split: str,
    reranker_fn: Callable,
    combine_fn: Callable,
    ctx: EvaluationContext,
    k: int = TOPK_FINAL,
) -> pd.DataFrame:
    """Run both orders and return a side-by-side metric comparison.

    Returns a DataFrame with columns: metric, order_a (rerank-then-combine),
    order_b (combine-then-rerank).
    """
    recs_a = rerank_then_combine(models, split, reranker_fn, k_rerank=k)
    recs_b = combine_then_rerank(models, split, combine_fn, reranker_fn, k_final=k)

    result_a = evaluate(recs_a, ctx, k=k, name="RerankThenCombine")
    result_b = evaluate(recs_b, ctx, k=k, name="CombineThenRerank")

    metrics = [
        "ndcg", "recall", "precision", "mrr", "map",
        "ild", "coverage", "novelty", "miscalibration",
        "avgpop", "tailshare", "gini", "entropy",
    ]
    rows = []
    for m in metrics:
        key = f"{m}@{k}"
        val_a = result_a.summary.get(key)
        val_b = result_b.summary.get(key)
        if val_a is not None and val_b is not None:
            rows.append({
                "metric": key,
                "rerank_then_combine": float(val_a),
                "combine_then_rerank": float(val_b),
                "difference": float(val_b) - float(val_a),
            })

    return pd.DataFrame(rows), recs_a, recs_b
