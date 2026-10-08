"""Exhaustive tuning for switching and mixed hybrids on the validation split (Task 1.5).

No model retraining is needed: each configuration just rearranges pre-loaded
recommendation rows and evaluates NDCG@10.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from project.hybrids.mixed import mixed_rrf, mixed_round_robin
from project.hybrids.switching import switching_recommendations
from project.metrics.evaluate import EvaluationContext, evaluate
from project.utils.data_formats import TOPK_FINAL

LEARNED_MODELS = ["ItemKNN", "UserKNN", "EASE", "BPR", "NeuMF", "LightGCN"]


def _ndcg(recs: pd.DataFrame, ctx: EvaluationContext) -> float:
    result = evaluate(recs, ctx, k=TOPK_FINAL, name="trial")
    return float(result.summary[f"ndcg@{TOPK_FINAL}"])


def tune_switching(
    candidate_models: Sequence[str] = LEARNED_MODELS,
    dimensions: Sequence[str] = ("activity", "taste"),
    split: str = "valid",
) -> Dict:
    """Try every assignment of candidate models to 3 groups, for each dimension.

    Returns dict with 'best_dimension', 'best_assignment', 'best_ndcg',
    and 'trials' (list of dicts).
    """
    ctx = EvaluationContext.for_split(split)
    groups_per_dim = {"activity": ["low", "medium", "high"],
                      "taste": ["niche", "mixed", "mainstream"]}

    trials = []
    best_ndcg = -1.0
    best_assignment = None
    best_dim = None

    for dim in dimensions:
        group_labels = groups_per_dim[dim]
        for combo in itertools.product(candidate_models, repeat=len(group_labels)):
            assignment = dict(zip(group_labels, combo))
            recs = switching_recommendations(assignment, split, dimension=dim)
            ndcg = _ndcg(recs, ctx)
            trials.append({"dimension": dim, **{f"model_{g}": m for g, m in assignment.items()},
                           "ndcg": ndcg})
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_assignment = assignment
                best_dim = dim

    return {
        "best_dimension": best_dim,
        "best_assignment": best_assignment,
        "best_ndcg": best_ndcg,
        "trials": pd.DataFrame(trials).sort_values("ndcg", ascending=False).reset_index(drop=True),
    }


def tune_mixed(
    candidate_models: Sequence[str] = LEARNED_MODELS,
    min_models: int = 2,
    max_models: int = 5,
    rrf_k_values: Sequence[int] = (20, 40, 60, 80),
    split: str = "valid",
) -> Dict:
    """Try model subsets of size min..max with both round-robin and RRF.

    Returns dict with 'best_method', 'best_models', 'best_rrf_k', 'best_ndcg',
    and 'trials' (list of dicts).
    """
    ctx = EvaluationContext.for_split(split)

    trials = []
    best_ndcg = -1.0
    best_config = {}

    subsets = []
    for size in range(min_models, max_models + 1):
        subsets.extend(itertools.combinations(candidate_models, size))

    for models in subsets:
        models_list = list(models)
        models_str = "+".join(models_list)

        recs_rr = mixed_round_robin(models_list, split)
        ndcg_rr = _ndcg(recs_rr, ctx)
        trials.append({"method": "round_robin", "models": models_str, "rrf_k": None, "ndcg": ndcg_rr})
        if ndcg_rr > best_ndcg:
            best_ndcg = ndcg_rr
            best_config = {"method": "round_robin", "models": models_list, "rrf_k": None}

        for rrf_k in rrf_k_values:
            recs_rrf = mixed_rrf(models_list, split, rrf_k=rrf_k)
            ndcg_rrf = _ndcg(recs_rrf, ctx)
            trials.append({"method": "rrf", "models": models_str, "rrf_k": rrf_k, "ndcg": ndcg_rrf})
            if ndcg_rrf > best_ndcg:
                best_ndcg = ndcg_rrf
                best_config = {"method": "rrf", "models": models_list, "rrf_k": rrf_k}

    return {
        "best_method": best_config.get("method"),
        "best_models": best_config.get("models"),
        "best_rrf_k": best_config.get("rrf_k"),
        "best_ndcg": best_ndcg,
        "trials": pd.DataFrame(trials).sort_values("ndcg", ascending=False).reset_index(drop=True),
    }
