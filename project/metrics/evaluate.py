"""One entry point for every metric (Task 2.1): ``evaluate(recs, context)``.

    from project.metrics import EvaluationContext, evaluate
    from project.utils.data_formats import load_recommendations

    ctx = EvaluationContext.for_split("test")      # ground truth, masked history, training data, genres, groups
    res = evaluate(load_recommendations("EASE", "test"), ctx)   # final list = rank <= 10
    res.summary         # pd.Series: every metric (macro-average over users, or global) + user counts
    res.per_user        # pd.DataFrame indexed by user_id: per-user values (group analyses, GRU, tests)
    res.item_exposure   # pd.Series: number of evaluated lists that contain each catalogue item

Any recommendation list in the shared format (``user_id, item_id, rank[, score]``) can be
evaluated: models, hybrids and re-rankers alike. RecBole is not used for any metric here; it
only produced the lists. Evaluation contract (see project/metrics/README.md):

* evaluated users U: every user with at least one relevant held-out item (all 943 by default);
  a user of U without a list counts as an empty list (0 accuracy, recorded in ``users_missing_list``);
  list users outside U are ignored (``users_ignored``).
* a list must not contain duplicates, unknown items or items of the user's history (that would
  mean the list belongs to the other split) -- these raise instead of being scored.
* per-user metrics are macro-averaged; coverage, Gini, entropy and group exposure are computed
  once over all evaluated lists. Undefined per-user values (e.g. ILD of a 1-item list) are NaN,
  left out of the mean, and counted in ``users_short_list`` / ``users_missing_list``.
* item popularity, the item / user groups and the users' genre and popularity profiles come from
  the training split only (frozen across models and splits, see ``lookups``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd

from project.metrics import accuracy, calibration, diversity, fairness, lookups, novelty, popularity
from project.utils.data_formats import EVAL_SPLITS, TOPK_FINAL, load_history, load_recommendations, load_split

# key -> (label for tables, True if higher is better); per-user keys get "@K" appended
METRIC_INFO = {
    "precision": ("Precision", True), "recall": ("Recall", True), "f1": ("F1", True), "hit": ("Hit", True),
    "ndcg": ("NDCG", True), "mrr": ("MRR", True), "map": ("MAP", True),
    "ild": ("ILD", True), "coverage": ("Coverage", True), "novelty": ("Novelty", True),
    "serendipity": ("Serendipity", True), "avgpop": ("AvgPop", False), "tailshare": ("TailShare", True),
    "miscalibration": ("MC$_{KL}$", False), "upd": ("UPD", False), "gru_ndcg": ("GRU", False),
    "gini": ("Gini", False), "entropy": ("Entropy", True),
}


@dataclass(frozen=True)
class EvaluationContext:
    """Everything the metrics need besides the lists, aligned to ``users`` (rows) and ``catalogue`` (columns)."""

    split: str
    users: np.ndarray  # (U,) evaluated user ids
    catalogue: np.ndarray  # (I,) item ids
    relevant: np.ndarray  # (U, I) bool, held-out relevant items T_u
    history: np.ndarray  # (U, I) bool, items masked from the lists (train; + valid when testing)
    train: np.ndarray  # (U, I) bool, training interactions: the users' profiles for calibration / UPD
    genres: np.ndarray  # (I, G) bool multi-hot genres
    genre_names: list
    counts: np.ndarray  # (I,) training interactions per item c_i (all users)
    n_train_users: int
    item_groups: np.ndarray  # (I,) lookups.HEAD / MID / TAIL / UNSEEN
    user_groups: pd.DataFrame  # index user_id; columns activity (low/medium/high), taste (niche/mixed/mainstream)
    min_rating: Optional[float] = None
    users_without_relevant: int = 0  # users with history but no relevant held-out item (not evaluated)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def n_items(self) -> int:
        return len(self.catalogue)

    @property
    def item_pos(self) -> Dict[str, int]:
        if "item_pos" not in self._cache:
            self._cache["item_pos"] = {item: i for i, item in enumerate(self.catalogue)}
        return self._cache["item_pos"]

    @property
    def p_gi(self) -> np.ndarray:
        if "p_gi" not in self._cache:
            self._cache["p_gi"] = calibration.item_genre_distribution(self.genres)
        return self._cache["p_gi"]

    def popularity_lists(self, k: int) -> np.ndarray:
        """(U, k) the popularity baseline's list per user (serendipity's 'expected' items): the k items
        with the most training interactions that are not masked for the user."""
        key = ("pop", k)
        if key not in self._cache:
            hist = [np.flatnonzero(row) for row in self.history]
            self._cache[key] = novelty.popularity_baseline_lists(self.counts, hist, k)
        return self._cache[key]

    @classmethod
    def build(cls, ground_truth: Mapping[str, Set[str]], history: Mapping[str, Set[str]],
              item_genres: Mapping[str, Sequence[str]], catalogue: Optional[Sequence[str]] = None,
              train: Optional[Mapping[str, Set[str]]] = None, split: str = "custom",
              min_rating: Optional[float] = None, head_share: float = lookups.HEAD_SHARE,
              tail_share: float = lookups.TAIL_SHARE) -> "EvaluationContext":
        """Context from plain dicts (used by ``for_split`` and by the unit tests). ``history`` is what
        is masked from the lists; ``train`` (default: ``history``) is what profiles and groups use."""
        train = history if train is None else train
        catalogue = np.asarray(list(item_genres) if catalogue is None else catalogue, dtype=str)
        pos = {item: i for i, item in enumerate(catalogue)}
        unknown = {i for items in list(ground_truth.values()) + list(history.values()) for i in items} - set(pos)
        if unknown:
            raise ValueError(f"{len(unknown)} items are not in the catalogue, e.g. {sorted(unknown)[:5]}")
        users = np.array(sorted(u for u, items in ground_truth.items() if items), dtype=str)
        relevant = np.zeros((len(users), len(catalogue)), dtype=bool)
        hist, train_m = np.zeros_like(relevant), np.zeros_like(relevant)
        for r, u in enumerate(users):
            relevant[r, [pos[i] for i in ground_truth[u]]] = True
            hist[r, [pos[i] for i in history.get(u, ())]] = True
            train_m[r, [pos[i] for i in train.get(u, ())]] = True
        if (relevant & hist).any():
            raise ValueError("held-out relevant items overlap the history -- ground truth and history do not match")
        if (train_m & ~hist).any():
            raise ValueError("training interactions must be part of the masked history")
        counts = lookups.item_counts(train, catalogue)
        item_groups = lookups.popularity_groups(counts, head_share, tail_share)
        genres, genre_names = lookups.genre_matrix(item_genres, catalogue)
        return cls(split=split, users=users, catalogue=catalogue, relevant=relevant, history=hist, train=train_m,
                   genres=genres, genre_names=genre_names, counts=counts, n_train_users=len(train),
                   item_groups=item_groups, user_groups=lookups.user_groups(train, item_groups, catalogue),
                   min_rating=min_rating,
                   users_without_relevant=len(set(history) | set(ground_truth)) - len(users))

    @classmethod
    def for_split(cls, split: str, min_rating: Optional[float] = None) -> "EvaluationContext":
        """Context of the frozen split: ``valid`` or ``test``. Relevance = every held-out interaction
        (the agreed protocol) or, with ``min_rating``, held-out interactions rated >= min_rating."""
        return _context_for_split(split, min_rating)


@lru_cache(maxsize=4)
def _context_for_split(split: str, min_rating: Optional[float]) -> EvaluationContext:
    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {EVAL_SPLITS}, got {split!r}")
    held_out = load_split(split)
    if min_rating is not None:
        held_out = held_out[held_out["rating"] >= min_rating]
    ground_truth = {u: set(items) for u, items in held_out.groupby("user_id")["item_id"]}
    train = {u: set(items) for u, items in load_split("train").groupby("user_id")["item_id"]}
    return EvaluationContext.build(ground_truth, load_history(split), lookups.load_item_genres(),
                                   lookups.load_catalogue(), train=train, split=split, min_rating=min_rating)


@dataclass
class EvaluationResult:
    name: str
    split: str
    k: int
    summary: pd.Series
    per_user: pd.DataFrame
    item_exposure: pd.Series


def index_matrix(recs: pd.DataFrame, ctx: EvaluationContext, k: int):
    """(U, k) catalogue indices of the top-k lists aligned to ``ctx.users`` (``-1`` = empty slot),
    plus the number of list users that are not evaluated."""
    missing = {"user_id", "item_id", "rank"} - set(recs.columns)
    if missing:
        raise ValueError(f"recommendation list is missing columns {sorted(missing)}")
    top = recs.loc[recs["rank"] <= k, ["user_id", "item_id", "rank"]].astype({"user_id": str, "item_id": str})
    for cols, what in ((["user_id", "item_id"], "the same item twice"), (["user_id", "rank"], "the same rank twice")):
        dup = top.duplicated(cols)
        if dup.any():
            raise ValueError(f"{dup.sum()} rows give a user {what}, e.g. {top[dup].iloc[0].to_dict()}")
    pos = ctx.item_pos
    unknown = ~top["item_id"].isin(pos)
    if unknown.any():
        raise ValueError(f"{unknown.sum()} recommended items are not in the catalogue, e.g. {top.item_id[unknown].iloc[0]}")
    row_of = pd.Series(np.arange(len(ctx.users)), index=ctx.users)
    evaluated = top["user_id"].isin(row_of.index)
    ignored = top.loc[~evaluated, "user_id"].nunique()
    top = top[evaluated].sort_values(["user_id", "rank"])
    rows = row_of[top["user_id"]].to_numpy()
    cols = top["item_id"].map(pos).to_numpy()
    idx = np.full((len(ctx.users), k), -1, dtype=np.int64)
    idx[rows, top.groupby("user_id").cumcount().to_numpy()] = cols
    leaked = ctx.history[rows, cols]
    if leaked.any():
        raise ValueError(f"{leaked.sum()} recommended items are in the user's history (masked items) -- "
                         f"is this a {ctx.split!r} list? e.g. user {top.user_id.iloc[np.argmax(leaked)]}")
    return idx, ignored


def evaluate(recs: pd.DataFrame, ctx: EvaluationContext, k: int = TOPK_FINAL, name: str = "") -> EvaluationResult:
    idx, ignored = index_matrix(recs, ctx, k)
    filled = idx >= 0
    safe = np.where(filled, idx, 0)
    hits = ctx.relevant[np.arange(len(idx))[:, None], safe] & filled
    n_rel = ctx.relevant.sum(axis=1)
    at = f"@{k}"

    per_user = pd.DataFrame({
        "precision" + at: accuracy.precision(hits),
        "recall" + at: accuracy.recall(hits, n_rel),
        "f1" + at: accuracy.f1(hits, n_rel),
        "hit" + at: accuracy.hit_rate(hits),
        "ndcg" + at: accuracy.ndcg(hits, n_rel),
        "mrr" + at: accuracy.reciprocal_rank(hits),
        "map" + at: accuracy.average_precision(hits, n_rel),
        "ild" + at: diversity.intra_list_diversity(idx, ctx.genres),
        "novelty" + at: novelty.novelty(idx, ctx.counts),
        "serendipity" + at: novelty.serendipity(hits, idx, ctx.popularity_lists(k)),
        "avgpop" + at: popularity.average_popularity(idx, ctx.counts, ctx.n_train_users),
        "tailshare" + at: popularity.group_share(idx, ctx.item_groups, [lookups.TAIL, lookups.UNSEEN]),
        "miscalibration" + at: calibration.miscalibration(idx, ctx.train, ctx.p_gi),
        "upd" + at: popularity.user_popularity_deviation(idx, ctx.train, ctx.item_groups,
                                                         len(lookups.POPULARITY_GROUPS)),
        "list_length": filled.sum(axis=1),
        "n_relevant": n_rel,
    }, index=pd.Index(ctx.users, name="user_id"))
    per_user[["activity", "taste"]] = ctx.user_groups.reindex(per_user.index)[["activity", "taste"]].values

    exposure = fairness.item_exposure(idx, ctx.n_items)
    gru = fairness.group_recommendation_unfairness(per_user["ndcg" + at], per_user["activity"], lookups.ACTIVITY_GROUPS)
    group_exp = fairness.group_exposure(idx, ctx.item_groups, len(lookups.POPULARITY_GROUPS))

    summary = {"name": name, "split": ctx.split, "k": k}
    summary.update(per_user.filter(like=at).mean())  # pandas mean skips NaN (undefined values are counted below)
    summary.update({
        "coverage" + at: diversity.catalogue_coverage(idx, ctx.n_items),
        "gini" + at: fairness.gini_index(exposure),
        "entropy" + at: fairness.shannon_entropy(exposure, normalised=True),
        "gru_ndcg" + at: gru["gru"],
        **{f"ndcg{at}_activity_{g}": v for g, v in gru["means"].items()},
        **{f"users_activity_{g}": v for g, v in gru["sizes"].items()},
        **{f"exposure_{g}" + at: v for g, v in zip(lookups.POPULARITY_GROUPS, group_exp)},
        "users": len(idx),
        "users_missing_list": int((~filled.any(axis=1)).sum()),
        "users_short_list": int((filled.any(axis=1) & ~filled.all(axis=1)).sum()),
        "users_ignored": int(ignored),
        "users_without_relevant": ctx.users_without_relevant,
        "min_rating": ctx.min_rating,
    })
    return EvaluationResult(name, ctx.split, k, pd.Series(summary, name=name), per_user,
                            pd.Series(exposure, index=pd.Index(ctx.catalogue, name="item_id"), name="exposure"))


def evaluate_saved(name: str, split: str, k: int = TOPK_FINAL, min_rating: Optional[float] = None) -> EvaluationResult:
    """Evaluate ``results/raw/recommendations/<name>_<split>_top50.csv``."""
    return evaluate(load_recommendations(name, split), EvaluationContext.for_split(split, min_rating), k, name)
