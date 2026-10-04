"""Popularity bias of the lists and user-side popularity calibration (W3S2 Fairness, slides 19, 39-40).

Item popularity c_i is the number of training interactions; the head / mid / tail / unseen
groups come from ``lookups.popularity_groups`` (head = most popular items holding 20 % of the
interactions, tail = least popular items holding 20 %, unseen = no training interaction).

    AvgPop@K    = mean_{i in L} c_i / |U|                 share of users who interacted with the item
    TailShare@K = |{i in L : i in tail or unseen}| / |L|   share of long-tail items in the list
    UPD(u)      = dist(P(H_u), P(L_u))                    slide 39, User Popularity Deviation

The slide leaves P and dist open. Our choice: P(.) = the distribution of a profile's (training
interactions) or list's items over the head / mid / tail / unseen groups, dist = Jensen-Shannon
divergence with log base 2 (symmetric, finite even when a group is missing on one side, in [0, 1]).
Lower UPD means the list follows the user's own taste for popular vs. niche films.
"""
from __future__ import annotations

import numpy as np

from project.metrics.novelty import mean_over_list


def average_popularity(idx: np.ndarray, counts: np.ndarray, n_users: int) -> np.ndarray:
    return mean_over_list(idx, np.asarray(counts, dtype=float) / n_users)


def group_share(idx: np.ndarray, groups: np.ndarray, group) -> np.ndarray:
    """Share of each list's items that belong to ``group`` (a code or a list of codes; NaN for an empty list)."""
    return mean_over_list(idx, np.isin(np.asarray(groups), group).astype(float))


def group_distribution_of_lists(idx: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """(U, n_groups) distribution of each list's items over the item groups (NaN rows if empty)."""
    return np.stack([group_share(idx, groups, g) for g in range(n_groups)], axis=1)


def group_distribution_of_histories(history: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """(U, n_groups) distribution of each (U, I) bool history over the item groups."""
    one_hot = np.eye(n_groups)[np.asarray(groups)]
    mass = history.astype(float) @ one_hot
    totals = mass.sum(axis=1, keepdims=True)
    return np.divide(mass, totals, out=np.full(mass.shape, np.nan), where=totals > 0)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence, log base 2 (in [0, 1]); NaN rows stay NaN."""
    m = 0.5 * (p + q)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a > 0, a * np.log2(a / b), 0.0).sum(axis=1)

    out = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    out[np.isnan(p).any(axis=1) | np.isnan(q).any(axis=1)] = np.nan
    return out


def user_popularity_deviation(idx: np.ndarray, history: np.ndarray, groups: np.ndarray, n_groups: int = 4) -> np.ndarray:
    return jensen_shannon(group_distribution_of_histories(history, groups, n_groups),
                          group_distribution_of_lists(idx, groups, n_groups))
