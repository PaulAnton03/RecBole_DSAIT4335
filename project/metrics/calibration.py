"""Genre miscalibration (W3S1 Calibration, slides 9-12; Steck, RecSys 2018).

Compares the genre mix of a user's history with the genre mix of the recommendation list
(this is calibration of the *list*, not of predicted probabilities).

    p(g|i)  item genre distribution. The slides show multi-hot genre vectors; KL needs a
            probability distribution, so an item with m genres puts 1/m on each of them.
    p(g|u)  = sum_{i in H_u} w_ui p(g|i) / sum w_ui        history profile, uniform w_ui = 1
    q(g|u)  = sum_{i in L_u} w_r(i) p(g|i) / sum w_r(i)     list profile, uniform w = 1
              (optional rank weights 1/log2(r + 1), the NDCG discount the slide mentions)
    q~      = (1 - alpha) q + alpha p,  alpha = 0.01        slide 12, avoids q = 0
    MC_KL   = KL(p || q~) = sum_{g: p(g|u) > 0} p(g|u) ln(p(g|u) / q~(g|u))

Natural logarithm (the slides do not fix the base). Lower is better, 0 = the list matches the
history's genre mix. The history H_u is the user's known interactions (never held-out ones).
Users with an empty history or an empty list are NaN (excluded from the mean and counted).

The re-rankers (Track E) reuse ``item_genre_distribution``, ``profile`` and ``kl_miscalibration``
so that they optimise exactly the quantity reported here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from project.metrics.accuracy import discounts

ALPHA = 0.01


def item_genre_distribution(genres: np.ndarray) -> np.ndarray:
    """(I, G) p(g|i): each item's genres weighted 1/m (rows sum to 1)."""
    g = genres.astype(float)
    return g / g.sum(axis=1, keepdims=True)


def profile(member: np.ndarray, p_gi: np.ndarray) -> np.ndarray:
    """(U, G) weighted genre profile from a (U, I) weight matrix (bool history or rank weights).

    Rows without any weight are NaN.
    """
    w = member.astype(float)
    totals = w.sum(axis=1, keepdims=True)
    mass = w @ p_gi
    return np.divide(mass, totals, out=np.full(mass.shape, np.nan), where=totals > 0)


def list_profile(idx: np.ndarray, p_gi: np.ndarray, rank_weighted: bool = False) -> np.ndarray:
    """(U, G) q(g|u) of top-K lists given as an index matrix (``-1`` = empty slot)."""
    filled = idx >= 0
    w = np.where(filled, discounts(idx.shape[1]) if rank_weighted else 1.0, 0.0)
    mass = (p_gi[np.where(filled, idx, 0)] * w[:, :, None]).sum(axis=1)
    totals = w.sum(axis=1, keepdims=True)
    return np.divide(mass, totals, out=np.full(mass.shape, np.nan), where=totals > 0)


def kl_miscalibration(p: np.ndarray, q: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """KL(p || (1 - alpha) q + alpha p) per row; NaN where p or q is NaN."""
    p, q = np.atleast_2d(p), np.atleast_2d(q)
    q_tilde = (1 - alpha) * q + alpha * p
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p / q_tilde), 0.0)
    out = terms.sum(axis=1)
    out[np.isnan(p).any(axis=1) | np.isnan(q).any(axis=1)] = np.nan
    return out


def miscalibration(idx: np.ndarray, history: np.ndarray, p_gi: np.ndarray, alpha: float = ALPHA,
                   rank_weighted: bool = False, p_history: Optional[np.ndarray] = None) -> np.ndarray:
    """MC_KL per user for lists ``idx`` (U, K) and a (U, I) bool history matrix."""
    p = profile(history, p_gi) if p_history is None else p_history
    return kl_miscalibration(p, list_profile(idx, p_gi, rank_weighted), alpha)
