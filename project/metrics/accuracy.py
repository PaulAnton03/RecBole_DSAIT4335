"""Accuracy metrics for top-K lists with binary relevance (Task 2.1).

Every function takes the hit matrix ``hits`` (U, K) bool -- ``hits[u, r]`` is True when the
item at rank ``r + 1`` of user u's list is a held-out relevant item -- plus, where needed,
``n_relevant`` (U,) = |T_u| >= 1, and returns one value per user; the caller macro-averages
over users. An empty slot of a short list is a non-hit, so short lists are never rewarded
(Precision always divides by K).

Definitions follow the lecture (W1S2 Evaluation 1, W2S1 Evaluation 2) with binary relevance
and are the same as RecBole's, which lets us check them against RecBole's numbers:

    Precision@K = |L ∩ T| / K                 Recall@K = |L ∩ T| / |T|
    F1@K        = 2PR / (P + R)  (0 if P = R = 0)
    Hit@K       = 1 if |L ∩ T| > 0
    NDCG@K      = sum_r rel_r / log2(r + 1)  /  sum_{r <= min(K, |T|)} 1 / log2(r + 1)
    RR@K        = 1 / rank of the first relevant item (0 if none)          -> MRR@K
    AP@K        = sum_r Precision@r * rel_r / min(K, |T|)                  -> MAP@K

AP normaliser: the lecture defines AP with "the number of relevant items", but its worked
example (Evaluation Part 2, slide 14) divides by the number of hits in the list (3 of 5).
We use min(K, |T|) -- RecBole's convention, so a user whose relevant items cannot all fit in
K slots is not punished for it.
"""
from __future__ import annotations

import numpy as np


def _check(hits: np.ndarray, n_relevant: np.ndarray | None = None):
    if hits.ndim != 2:
        raise ValueError("hits must be a (users, K) matrix")
    if n_relevant is not None and (np.asarray(n_relevant) < 1).any():
        raise ValueError("every evaluated user needs at least one relevant item (|T_u| >= 1)")


def discounts(k: int) -> np.ndarray:
    """1 / log2(r + 1) for ranks r = 1..k."""
    return 1.0 / np.log2(np.arange(2, k + 2))


def precision(hits: np.ndarray) -> np.ndarray:
    _check(hits)
    return hits.sum(axis=1) / hits.shape[1]


def recall(hits: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    _check(hits, n_relevant)
    return hits.sum(axis=1) / n_relevant


def f1(hits: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    p, r = precision(hits), recall(hits, n_relevant)
    denom = p + r
    return np.divide(2 * p * r, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def hit_rate(hits: np.ndarray) -> np.ndarray:
    _check(hits)
    return hits.any(axis=1).astype(float)


def ndcg(hits: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    _check(hits, n_relevant)
    disc = discounts(hits.shape[1])
    dcg = (hits * disc).sum(axis=1)
    idcg = np.cumsum(disc)[np.minimum(n_relevant, hits.shape[1]) - 1]
    return dcg / idcg


def reciprocal_rank(hits: np.ndarray) -> np.ndarray:
    _check(hits)
    first = hits.argmax(axis=1)  # index of the first True (0 if none, masked below)
    return np.where(hits.any(axis=1), 1.0 / (first + 1), 0.0)


def average_precision(hits: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    _check(hits, n_relevant)
    k = hits.shape[1]
    precision_at_r = np.cumsum(hits, axis=1) / np.arange(1, k + 1)
    return (precision_at_r * hits).sum(axis=1) / np.minimum(n_relevant, k)
