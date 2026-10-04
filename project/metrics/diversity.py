"""Diversity: intra-list diversity and catalogue coverage (W2S2 Diversity, slides 9 and 12).

Lists are given as an index matrix ``idx`` (U, K) int into the catalogue, ``-1`` = empty slot.

ILD(L) = 1 / (|L| (|L| - 1)) * sum_{i != j in L} d(i, j)   (slide 9, ordered pairs)

which, for a symmetric distance, is the mean distance over the |L|(|L|-1)/2 unordered pairs.
The slide leaves d open ("genre/topic/category overlap" or "embedding distance"); we use the
Jaccard distance between genre sets, d(i, j) = 1 - |G_i ∩ G_j| / |G_i ∪ G_j|, for every model
alike -- a model's own embeddings would change the yardstick between models. ILD is undefined
(NaN, excluded from the mean and counted) for lists with fewer than two items.

Coverage@K = |union_u L_u| / |I|  (slide 12): one global value over all evaluated users.
"""
from __future__ import annotations

import numpy as np


def jaccard_distance(features: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Jaccard distance between the feature sets of items ``a`` and ``b`` (same shape, any)."""
    fa, fb = features[a], features[b]
    inter = (fa & fb).sum(axis=-1)
    union = (fa | fb).sum(axis=-1)
    return 1.0 - inter / union  # union > 0: lookups.genre_matrix guarantees >= 1 genre per item


def jaccard_distance_matrix(features: np.ndarray) -> np.ndarray:
    """(I, I) Jaccard distances between all items -- for re-rankers (MMR) that need many lookups."""
    f = features.astype(np.float32)
    inter = f @ f.T
    sizes = f.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    return 1.0 - inter / union


def intra_list_diversity(idx: np.ndarray, features: np.ndarray) -> np.ndarray:
    """ILD per user (NaN for lists with fewer than two items)."""
    k = idx.shape[1]
    a, b = np.triu_indices(k, 1)  # all unordered position pairs
    ia, ib = idx[:, a], idx[:, b]
    both = (ia >= 0) & (ib >= 0)
    d = jaccard_distance(features, np.where(both, ia, 0), np.where(both, ib, 0))
    n_pairs = both.sum(axis=1)
    total = np.where(both, d, 0.0).sum(axis=1)
    return np.divide(total, n_pairs, out=np.full(len(idx), np.nan), where=n_pairs > 0)


def catalogue_coverage(idx: np.ndarray, n_items: int) -> float:
    """Fraction of the catalogue recommended to at least one user."""
    return len(np.unique(idx[idx >= 0])) / n_items
