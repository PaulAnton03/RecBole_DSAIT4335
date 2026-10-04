"""Shared item and user lookups (Track B): catalogue, genres, popularity and group labels.

Every other track that needs movie genres, movie popularity or the user / item group labels
imports them from here, so the re-rankers optimise exactly what the metrics measure.

Leakage rule: everything derived from interactions (popularity, popularity groups, user
activity, user genre profiles) is computed from the user histories that are *known* when a
split is evaluated -- ``data_formats.load_history(split)``: train for ``valid``, train + valid
for ``test`` (exactly the items RecBole masks). Held-out interactions are never used.
"""
from __future__ import annotations

from typing import Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from project.utils.data_formats import DATASET_NAME
from project.utils.paths import DATASET_DIR

ITEM_FILE = DATASET_DIR / DATASET_NAME / f"{DATASET_NAME}.item"

# popularity groups (codes are used in the per-item arrays)
HEAD, MID, TAIL = 0, 1, 2
POPULARITY_GROUPS = ("head", "mid", "tail")
HEAD_SHARE = 0.2  # head = most popular items that together hold >= 20 % of the interactions
TAIL_SHARE = 0.2  # tail = least popular items that together hold <= 20 % of the interactions

ACTIVE_FRACTION = 0.2  # "active" users = the 20 % with the largest histories (GRU groups)


def load_item_genres() -> pd.Series:
    """Genres of every movie in the catalogue: ``item_id`` (str) -> tuple of genre names.

    Taken from the ``class`` field of ``dataset/ml-100k/ml-100k.item`` (19 genres). The two
    movies labelled ``unknown`` keep it as their genre, so every item has at least one genre
    and no item-item distance is undefined.
    """
    df = pd.read_csv(ITEM_FILE, sep="\t", dtype=str)
    df.columns = [c.split(":")[0] for c in df.columns]
    genres = df["class"].fillna("").str.split()
    return pd.Series([tuple(g) for g in genres], index=df["item_id"].values, name="genres")


def load_catalogue() -> np.ndarray:
    """The evaluable catalogue I: all 1,682 movies of ML-100K (every one is a candidate for
    some user; there are no padding or reserved ids in the exported files)."""
    return load_item_genres().index.to_numpy(dtype=str)


def genre_matrix(item_genres: Mapping[str, Sequence[str]], catalogue: Sequence[str]) -> Tuple[np.ndarray, list]:
    """Multi-hot (I, G) bool matrix in catalogue order and the genre names (sorted)."""
    names = sorted({g for item in catalogue for g in item_genres[item]})
    col = {g: j for j, g in enumerate(names)}
    m = np.zeros((len(catalogue), len(names)), dtype=bool)
    for i, item in enumerate(catalogue):
        for g in item_genres[item]:
            m[i, col[g]] = True
    if not m.any(axis=1).all():
        missing = [catalogue[i] for i in np.flatnonzero(~m.any(axis=1))[:5]]
        raise ValueError(f"every catalogue item needs at least one genre; none for e.g. {missing}")
    return m, names


def item_counts(history: Mapping[str, Set[str]], catalogue: Sequence[str]) -> np.ndarray:
    """c_i: number of known (history) interactions of every catalogue item; 0 if never seen."""
    pos = {item: i for i, item in enumerate(catalogue)}
    counts = np.zeros(len(catalogue), dtype=np.int64)
    for items in history.values():
        for item in items:
            counts[pos[item]] += 1
    return counts


def popularity_groups(counts: np.ndarray, head_share: float = HEAD_SHARE, tail_share: float = TAIL_SHARE) -> np.ndarray:
    """Head / mid / tail code (``HEAD``, ``MID``, ``TAIL``) for every item by its interaction count.

    Head: the most popular items that together account for at least ``head_share`` of all
    interactions; tail: the least popular items that together account for at most ``tail_share``
    (this includes items without interactions); mid: the rest. Items with equal counts always
    fall in the same group (the cut is a count threshold), so the shares are approximate.
    """
    counts = np.asarray(counts)
    total = counts.sum()
    values = np.unique(counts)  # ascending distinct counts
    mass_at_least = np.array([counts[counts >= v].sum() for v in values])
    mass_at_most = np.array([counts[counts <= v].sum() for v in values])
    head_min = values[mass_at_least >= head_share * total].max()  # largest threshold reaching the share
    tail_ok = values[mass_at_most <= tail_share * total]
    groups = np.full(len(counts), MID, dtype=np.int8)
    groups[counts >= head_min] = HEAD
    if len(tail_ok):
        groups[(counts <= tail_ok.max()) & (groups != HEAD)] = TAIL
    return groups


def user_activity_groups(history: Mapping[str, Set[str]], active_fraction: float = ACTIVE_FRACTION) -> pd.Series:
    """``active`` for the ``active_fraction`` of users with the largest histories (ties at the
    threshold included), ``inactive`` for the rest. Used as the two user groups of GRU."""
    sizes = pd.Series({u: len(items) for u, items in history.items()})
    threshold = sizes.quantile(1 - active_fraction)
    return pd.Series(np.where(sizes >= threshold, "active", "inactive"), index=sizes.index, name="activity")


def describe_popularity_groups(counts: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Size, interaction share and count range of each popularity group (for the report)."""
    rows = []
    for code, name in enumerate(POPULARITY_GROUPS):
        c = counts[groups == code]
        rows.append({"group": name, "items": int(len(c)), "interaction_share": float(c.sum() / counts.sum()),
                     "min_count": int(c.min()) if len(c) else 0, "max_count": int(c.max()) if len(c) else 0})
    return pd.DataFrame(rows)
