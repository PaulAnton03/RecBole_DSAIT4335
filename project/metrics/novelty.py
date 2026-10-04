"""Novelty and serendipity (W2S2 Diversity, slides 14-17).

Novelty (slide 15, self-information): novelty(i) = -log2 pop(i), with pop(i) = c_i / sum_j c_j
the share of all known interactions that are on item i. The slide defines novelty per item;
a list scores the mean over its items, users are macro-averaged. Items without any known
interaction (pop = 0, infinite surprisal) get the count 1, i.e. the novelty of the rarest
observed item, rather than an unbounded or invented value.

Serendipity (slides 16-17: "unexpected and useful"; unexpectedness measured against a baseline
recommender such as popularity). The slide gives no formula, so we choose the common
baseline-based one: an item is serendipitous when it is relevant (a held-out item) AND not in the
list the popularity baseline would show this user (the K most popular items outside the user's
history). Serendipity@K = |{i in L : i in T, i not in Pop_u@K}| / K. Pop itself scores 0.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def self_information(counts: np.ndarray, min_count: float = 1.0) -> np.ndarray:
    """-log2 pop(i) for every catalogue item, pop(i) = max(c_i, min_count) / sum_j c_j."""
    counts = np.asarray(counts, dtype=float)
    return -np.log2(np.maximum(counts, min_count) / counts.sum())


def mean_over_list(idx: np.ndarray, item_values: np.ndarray) -> np.ndarray:
    """Mean of a per-item value over each list's items (NaN for an empty list)."""
    filled = idx >= 0
    vals = np.where(filled, item_values[np.where(filled, idx, 0)], 0.0)
    n = filled.sum(axis=1)
    return np.divide(vals.sum(axis=1), n, out=np.full(len(idx), np.nan), where=n > 0)


def novelty(idx: np.ndarray, counts: np.ndarray, min_count: float = 1.0) -> np.ndarray:
    return mean_over_list(idx, self_information(counts, min_count))


def popularity_baseline_lists(counts: np.ndarray, history_idx: Sequence[np.ndarray], k: int) -> np.ndarray:
    """(U, k) index matrix: the k most popular items outside each user's history (ties -> catalogue order)."""
    order = np.argsort(-np.asarray(counts), kind="stable")
    out = np.full((len(history_idx), k), -1, dtype=np.int64)
    for u, hist in enumerate(history_idx):
        seen = set(hist.tolist())
        picked = [i for i in order[: k + len(seen)] if i not in seen][:k]
        out[u, : len(picked)] = picked
    return out


def serendipity(hits: np.ndarray, idx: np.ndarray, baseline_idx: np.ndarray) -> np.ndarray:
    """Share of the K slots holding a relevant item that the baseline list does not contain."""
    expected = (idx[:, :, None] == baseline_idx[:, None, :]).any(axis=2) & (idx >= 0)
    return (hits & ~expected).sum(axis=1) / idx.shape[1]
