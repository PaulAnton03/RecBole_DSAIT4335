"""Shared item and user lookups (Track B): catalogue, genres, popularity and group labels.

Every other track that needs movie genres, movie popularity or the user / item group labels
imports them from here, so the re-rankers optimise exactly what the metrics measure.

Leakage rule: everything derived from interactions (item popularity, the item and user groups,
user genre and popularity profiles) is computed from the TRAINING split only -- the data every
model was trained on -- so it is identical for ``valid`` and ``test`` and frozen across models.
Held-out interactions are never used; the validation interactions are only masked from the lists
when ``test`` is evaluated (``data_formats.load_history``).
"""
from __future__ import annotations

from typing import Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from project.utils.data_formats import DATASET_NAME
from project.utils.paths import DATASET_DIR

ITEM_FILE = DATASET_DIR / DATASET_NAME / f"{DATASET_NAME}.item"

# item popularity groups (codes are used in the per-item arrays)
HEAD, MID, TAIL, UNSEEN = 0, 1, 2, 3
POPULARITY_GROUPS = ("head", "mid", "tail", "unseen")
HEAD_SHARE = 0.2  # head = most popular items that together hold >= 20 % of the training interactions
TAIL_SHARE = 0.2  # tail = least popular items that together hold <= 20 % of the training interactions

# user groups: tertiles of a per-user score computed on the training split
ACTIVITY_GROUPS = ("low", "medium", "high")  # number of training interactions
TASTE_GROUPS = ("niche", "mixed", "mainstream")  # share of the user's training interactions on head items


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
    """Head / mid / tail / unseen code (``HEAD``, ``MID``, ``TAIL``, ``UNSEEN``) for every item.

    Items sorted by interaction count: head = the most popular items that together account for at
    least ``head_share`` of all interactions; tail = the least popular items that together account
    for at most ``tail_share``; mid = the rest. Items without any interaction are kept apart as
    ``UNSEEN`` (never observed, not merely rare). Items with equal counts always fall in the same
    group (the cuts are count thresholds), so the shares are approximate.
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
    groups[counts == 0] = UNSEEN
    return groups


def tertile_groups(scores: pd.Series, labels: Sequence[str]) -> pd.Series:
    """Three groups by the 1/3 and 2/3 quantiles of ``scores`` (low <= t1 < medium <= t2 < high).

    The cuts are value thresholds, so users with equal scores always share a group; sizes are
    therefore only approximately equal.
    """
    t1, t2 = scores.quantile([1 / 3, 2 / 3])
    out = np.where(scores <= t1, labels[0], np.where(scores <= t2, labels[1], labels[2]))
    return pd.Series(out, index=scores.index)


def user_activity_scores(train: Mapping[str, Set[str]]) -> pd.Series:
    """Number of training interactions per user."""
    return pd.Series({u: len(items) for u, items in train.items()}, dtype=float)


def user_head_shares(train: Mapping[str, Set[str]], item_groups: np.ndarray, catalogue: Sequence[str]) -> pd.Series:
    """Share of each user's training interactions that are on head items."""
    head = {item for item, g in zip(catalogue, item_groups) if g == HEAD}
    return pd.Series({u: sum(i in head for i in items) / len(items) for u, items in train.items() if items})


def user_groups(train: Mapping[str, Set[str]], item_groups: np.ndarray, catalogue: Sequence[str]) -> pd.DataFrame:
    """Frozen user groups (index user_id): ``activity`` (low / medium / high number of training
    interactions) and ``taste`` (niche / mixed / mainstream share of head items in the training
    interactions; the fairness lecture's users interested in unpopular, both, or popular items).
    The two dimensions are kept separate: an active user can still have niche taste."""
    return pd.DataFrame({
        "activity": tertile_groups(user_activity_scores(train), ACTIVITY_GROUPS),
        "taste": tertile_groups(user_head_shares(train, item_groups, catalogue), TASTE_GROUPS),
    }).rename_axis("user_id")


def group_definitions(train: Mapping[str, Set[str]], counts: np.ndarray, item_groups: np.ndarray,
                      catalogue: Sequence[str]) -> dict:
    """Thresholds and sizes of every group (written to results/processed/group_definitions.json)."""
    out = {"items": describe_popularity_groups(counts, item_groups).to_dict(orient="records"), "users": {}}
    groups = user_groups(train, item_groups, catalogue)
    for dim, scores in (("activity", user_activity_scores(train)),
                        ("taste", user_head_shares(train, item_groups, catalogue))):
        out["users"][dim] = [{"group": g, "users": int((groups[dim] == g).sum()),
                              "min_score": float(scores[groups[dim] == g].min()),
                              "max_score": float(scores[groups[dim] == g].max())}
                             for g in (ACTIVITY_GROUPS if dim == "activity" else TASTE_GROUPS)]
    return out


def describe_popularity_groups(counts: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Size, interaction share and count range of each item popularity group (for the report)."""
    rows = []
    for code, name in enumerate(POPULARITY_GROUPS):
        c = counts[groups == code]
        rows.append({"group": name, "items": int(len(c)), "interaction_share": float(c.sum() / counts.sum()),
                     "min_count": int(c.min()) if len(c) else 0, "max_count": int(c.max()) if len(c) else 0})
    return pd.DataFrame(rows)
