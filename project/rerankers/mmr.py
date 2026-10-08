"""Minimal MMR placeholder for Task 3.3 order comparison. Replace with Person E's rerankers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from project.metrics.diversity import jaccard_distance_matrix
from project.metrics.lookups import genre_matrix, load_catalogue, load_item_genres
from project.utils.data_formats import RECOMMENDATION_COLUMNS, TOPK_CANDIDATES, TOPK_FINAL

_DIST = None
_POS = None


def _ensure_dist():
    global _DIST, _POS
    if _DIST is None:
        genres_bool, _ = genre_matrix(load_item_genres(), load_catalogue())
        _DIST = jaccard_distance_matrix(genres_bool)
        _POS = {item: i for i, item in enumerate(load_catalogue())}


def mmr_rerank(recs, lam=0.5, k_out=TOPK_FINAL, k_in=TOPK_CANDIDATES):
    _ensure_dist()
    cands = recs[recs["rank"] <= k_in].sort_values(["user_id", "rank"])
    rows = []
    for uid, grp in cands.groupby("user_id"):
        items, scores = list(grp["item_id"]), list(grp["score"])
        s_lo, s_hi = min(scores), max(scores)
        rng = s_hi - s_lo or 1.0
        rel = [(s - s_lo) / rng for s in scores]
        alive, sel = set(range(len(items))), []
        for rank in range(1, k_out + 1):
            best_i, best_v = -1, -np.inf
            for i in alive:
                ci = _POS.get(items[i])
                div = 1.0 if (not sel or ci is None) else min(_DIST[ci, s] for s in sel)
                v = lam * rel[i] + (1 - lam) * div
                if v > best_v:
                    best_i, best_v = i, v
            if best_i < 0:
                break
            alive.discard(best_i)
            ci = _POS.get(items[best_i])
            if ci is not None:
                sel.append(ci)
            rows.append((uid, items[best_i], rank, scores[best_i]))
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
