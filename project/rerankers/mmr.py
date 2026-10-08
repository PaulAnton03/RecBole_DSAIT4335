"""MMR (Maximal Marginal Relevance) diversification reranker.

Greedy selection: at each step, pick the candidate item that maximizes
    lambda * relevance(i) + (1 - lambda) * diversity(i, already_selected)

where diversity is measured as the minimum Jaccard genre distance to the
already-selected items (maximum dissimilarity to the most similar pick).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from project.metrics.diversity import jaccard_distance_matrix
from project.metrics.evaluate import EvaluationContext
from project.metrics.lookups import genre_matrix, load_catalogue, load_item_genres
from project.utils.data_formats import (
    RECOMMENDATION_COLUMNS,
    TOPK_CANDIDATES,
    TOPK_FINAL,
)


def _build_dist_matrix() -> tuple:
    """Build the item-item Jaccard distance matrix and item-to-index mapping."""
    item_genres = load_item_genres()
    catalogue = load_catalogue()
    genres_bool, _ = genre_matrix(item_genres, catalogue)
    dist = jaccard_distance_matrix(genres_bool)
    item_pos = {item: i for i, item in enumerate(catalogue)}
    return dist, item_pos


def mmr_rerank(
    recs: pd.DataFrame,
    lam: float = 0.5,
    k_out: int = TOPK_FINAL,
    k_in: int = TOPK_CANDIDATES,
    dist_matrix: Optional[np.ndarray] = None,
    item_pos: Optional[dict] = None,
) -> pd.DataFrame:
    """Apply MMR diversification to a recommendation list.

    Parameters
    ----------
    recs : DataFrame with columns user_id, item_id, rank, score
        The candidate list (typically top-50 from a model).
    lam : float
        Trade-off parameter. lam=1 is pure relevance, lam=0 is pure diversity.
    k_out : int
        Number of items to select per user (default 10).
    k_in : int
        Maximum candidate rank to consider (default 50).
    dist_matrix : (I, I) array, optional
        Precomputed Jaccard distance matrix. Built on first call if not provided.
    item_pos : dict, optional
        Item-id to catalogue-index mapping. Built with dist_matrix if not provided.

    Returns
    -------
    DataFrame with columns user_id, item_id, rank, score (k_out rows per user).
    """
    if dist_matrix is None or item_pos is None:
        dist_matrix, item_pos = _build_dist_matrix()

    candidates = recs[recs["rank"] <= k_in].copy()

    rows = []
    for uid, grp in candidates.sort_values("rank").groupby("user_id"):
        items = list(grp["item_id"])
        scores_raw = list(grp["score"])

        if not items:
            continue

        s_min, s_max = min(scores_raw), max(scores_raw)
        s_range = s_max - s_min if s_max > s_min else 1.0
        rel = [(s - s_min) / s_range for s in scores_raw]

        cat_indices = [item_pos.get(iid) for iid in items]

        selected_cat = []
        selected_items = []
        remaining = list(range(len(items)))

        for out_rank in range(1, k_out + 1):
            if not remaining:
                break

            best_idx = None
            best_score = -np.inf

            for idx in remaining:
                ci = cat_indices[idx]
                relevance = rel[idx]

                if not selected_cat or ci is None:
                    diversity = 1.0
                else:
                    diversity = min(dist_matrix[ci, sc] for sc in selected_cat)

                mmr_score = lam * relevance + (1 - lam) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            remaining.remove(best_idx)
            selected_items.append(best_idx)
            if cat_indices[best_idx] is not None:
                selected_cat.append(cat_indices[best_idx])

            rows.append((uid, items[best_idx], out_rank, scores_raw[best_idx]))

    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
