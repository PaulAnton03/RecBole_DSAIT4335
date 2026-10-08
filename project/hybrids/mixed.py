"""Mixed hybrid: interleave top items from multiple models (Task 1.4).

Two methods are provided:
- Round-robin interleaving with Borda scores
- Reciprocal Rank Fusion (RRF)
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from project.utils.data_formats import (
    RECOMMENDATION_COLUMNS,
    TOPK_CANDIDATES,
    load_recommendations,
)

DEFAULT_MODELS = ["EASE", "ItemKNN", "BPR", "NeuMF"]
DEFAULT_RRF_K = 60


def mixed_round_robin(
    models: List[str],
    split: str,
    k: int = TOPK_CANDIDATES,
) -> pd.DataFrame:
    """Interleave models' top items in round-robin order, skipping duplicates.

    Each item's score is a Borda count: ``k - new_rank + 1``.
    """
    model_recs = {m: load_recommendations(m, split, k) for m in models}

    per_user_lists = {}
    for m, recs in model_recs.items():
        for uid, grp in recs.sort_values("rank").groupby("user_id"):
            per_user_lists.setdefault(uid, {})[m] = list(grp["item_id"])

    rows = []
    for uid, model_items in per_user_lists.items():
        seen = set()
        merged = []
        pointers = {m: 0 for m in models}
        while len(merged) < k:
            added_this_round = False
            for m in models:
                items = model_items.get(m, [])
                while pointers[m] < len(items) and items[pointers[m]] in seen:
                    pointers[m] += 1
                if pointers[m] < len(items):
                    item = items[pointers[m]]
                    pointers[m] += 1
                    seen.add(item)
                    merged.append(item)
                    added_this_round = True
                    if len(merged) >= k:
                        break
            if not added_this_round:
                break

        for rank, item in enumerate(merged, start=1):
            rows.append((uid, item, rank, float(k - rank + 1)))

    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def mixed_rrf(
    models: List[str],
    split: str,
    k: int = TOPK_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
) -> pd.DataFrame:
    """Reciprocal Rank Fusion: ``score(item) = sum_m 1 / (rrf_k + rank_m(item))``."""
    model_recs = {m: load_recommendations(m, split, k) for m in models}

    per_user_scores = {}
    for m, recs in model_recs.items():
        for _, row in recs.iterrows():
            uid = row["user_id"]
            iid = row["item_id"]
            rrf_score = 1.0 / (rrf_k + row["rank"])
            per_user_scores.setdefault(uid, {}).setdefault(iid, 0.0)
            per_user_scores[uid][iid] += rrf_score

    rows = []
    for uid, item_scores in per_user_scores.items():
        sorted_items = sorted(item_scores.items(), key=lambda x: -x[1])[:k]
        for rank, (iid, score) in enumerate(sorted_items, start=1):
            rows.append((uid, iid, rank, score))

    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
