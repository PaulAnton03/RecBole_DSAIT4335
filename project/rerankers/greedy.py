"""The shared greedy re-ranking loop (Task 3.1, Track E).

Every re-ranker takes a user's candidate list (Track A's ``<Name>_<split>_top50.csv``) and fills the
new list one slot at a time (W2S2 Diversity slides 10-11, W3S1 Calibration slides 13-18, W3S2
Fairness slide 48). With L the items placed so far, the next slot gets

    argmax_{i not in L}  (1 - lam) * rel(i) + lam * gain(i | L)

    rel(i)       the model's score, min-max normalised over the user's candidates to [0, 1].
                 Choice: the slides add raw scores; normalising per user makes one lam grid mean the
                 same for EASE, LightGCN or a hybrid, whose scores live on different scales.
    gain(i | L)  how much adding i improves the list's beyond-accuracy objective (``objectives.py``).
    lam          0 = the model's own order, 1 = the objective alone.

Convention: lam weights the beyond-accuracy term, as on the diversity and calibration slides (the
fairness slide writes the same trade-off with lam on relevance). Ties go to the candidate with the
better original rank, so lam = 0 returns the input list unchanged.

The loop is sequential, so the first k slots do not depend on how many slots are filled: ``depth``
can stop the greedy part early (a lam sweep that scores the top 10 only needs ``depth=10``) and the
remaining candidates keep their original order below it.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol

import numpy as np
import pandas as pd

from project.metrics import EvaluationContext
from project.utils.data_formats import RECOMMENDATION_COLUMNS, TOPK_CANDIDATES, TOPK_FINAL


class UserState(Protocol):
    """One user's objective while the list is being built."""

    def gains(self) -> np.ndarray:
        """(n,) gain(i | L) of every candidate given the items placed so far (placed ones are ignored)."""

    def add(self, j: int) -> None:
        """Candidate ``j`` (position in the candidate list) was placed in the next slot."""


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1] over one user's candidates; all-equal scores give 1 everywhere."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("candidate scores must be finite")
    lo, hi = scores.min(), scores.max()
    return np.ones_like(scores) if hi == lo else (scores - lo) / (hi - lo)


def greedy_order(rel: np.ndarray, state: UserState, lam: float, depth: Optional[int] = None) -> np.ndarray:
    """Candidate positions in their new order; ``rel`` is given in the original rank order."""
    n = len(rel)
    depth = n if depth is None else min(depth, n)
    free = np.ones(n, dtype=bool)
    order = []
    for _ in range(depth):
        value = (1 - lam) * rel + lam * state.gains()
        value[~free] = -np.inf
        j = int(np.argmax(value))  # first maximum = best original rank among ties
        order.append(j)
        free[j] = False
        state.add(j)
    return np.concatenate([np.asarray(order, dtype=np.int64), np.flatnonzero(free)])


def rerank(recs: pd.DataFrame, objective, lam: float, ctx: EvaluationContext,
           depth: Optional[int] = None) -> pd.DataFrame:
    """Re-rank every user's candidates in ``recs`` (shared format ``user_id, item_id, rank, score``).

    ``ctx`` is the evaluation context of the same split (genres, training profiles, item groups).
    Returns the same candidates per user in the new order with ``rank`` 1..n and
    ``score = (n - rank + 1) / n``. Choice: the score follows the new order, so lists built from
    re-ranked lists (Task 3.3) see the re-ranking; the model's own scores would undo it.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    missing = set(RECOMMENDATION_COLUMNS) - set(recs.columns)
    if missing:
        raise ValueError(f"recommendation list is missing columns {sorted(missing)}")
    recs = recs.astype({"user_id": str, "item_id": str}).sort_values(["user_id", "rank"], kind="stable")
    row_of = {u: r for r, u in enumerate(ctx.users)}
    pos = ctx.item_pos
    unknown_users = set(recs["user_id"]) - set(row_of)
    if unknown_users:
        raise ValueError(f"{len(unknown_users)} users are not in the {ctx.split!r} context, e.g. {sorted(unknown_users)[:3]}")
    unknown_items = ~recs["item_id"].isin(pos)
    if unknown_items.any():
        raise ValueError(f"{unknown_items.sum()} items are not in the catalogue, e.g. {recs.item_id[unknown_items].iloc[0]}")

    objective.prepare(ctx)
    users, items, ranks, scores = [], [], [], []
    for user, cand in recs.groupby("user_id", sort=False):
        cand_items = cand["item_id"].to_numpy()
        state = objective.user_state(row_of[user], cand["item_id"].map(pos).to_numpy())
        order = greedy_order(normalise_scores(cand["score"].to_numpy()), state, lam, depth)
        n = len(order)
        users.append(np.full(n, user, dtype=object))
        items.append(cand_items[order])
        ranks.append(np.arange(1, n + 1))
        scores.append((n - np.arange(n)) / n)
    return pd.DataFrame({"user_id": np.concatenate(users), "item_id": np.concatenate(items),
                         "rank": np.concatenate(ranks), "score": np.concatenate(scores)})


def list_reranker(objective, lam: float, ctx: EvaluationContext) -> Callable[..., pd.DataFrame]:
    """``rerank`` as a plain list-to-list function ``fn(recs, k_out, k_in)``, the interface of the
    pipeline-order comparison (Task 3.3, ``project/analysis/order_comparison.py``): re-ranks each
    user's top ``k_in`` candidates and returns the new top ``k_out``. ``ctx`` must be the context of
    the split the lists belong to; lam follows this module's convention (0 = the list's own order)."""

    def fn(recs: pd.DataFrame, k_out: int = TOPK_FINAL, k_in: int = TOPK_CANDIDATES) -> pd.DataFrame:
        out = rerank(recs[recs["rank"] <= k_in], objective, lam, ctx, depth=k_out)
        return out[out["rank"] <= k_out].reset_index(drop=True)

    return fn


def reranked_name(base: str, objective) -> str:
    """Stable list name of a re-ranked model, e.g. ``EASE+MMR`` (file ``EASE+MMR_<split>_top50.csv``)."""
    return f"{base}+{objective.name}"
