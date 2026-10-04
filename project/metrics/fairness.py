"""User- and item-side fairness (W3S2 Fairness, slides 38, 42, 43).

User side -- Group Recommendation Unfairness (slide 38, Fu et al. SIGIR 2020):

    GRU(G1, G2) = | mean_{u in G1} F(u) - mean_{u in G2} F(u) |

with F a per-user quality metric (we use NDCG@K, the slide's example). Lower = smaller quality
gap; always reported with both group means and sizes, since a small gap alone does not mean
good quality.

Item side -- equality of exposure (slide 42: "Gini index or entropy ... measure the flatness"):
the exposure of item i is e_i = number of evaluated lists that contain it, over the WHOLE
catalogue (never-recommended items count with e_i = 0). With n = |I| and e sorted ascending,

    Gini    = sum_j (2j - n - 1) e_(j) / (n sum_j e_j)     0 = equal exposure, max (n - 1) / n
    Entropy = -sum_{i: s_i > 0} s_i ln s_i,  s_i = e_i / sum_j e_j
    normalised entropy = Entropy / ln n                    1 = equal exposure

The slide gives no formulas; this Gini is the standard one (identical to RecBole's GiniIndex).
Both are global: computed once over all evaluated lists, never per batch.

Demographic parity of exposure between item groups (slide 43): the mean position-discounted
exposure an item of group G_k receives per list, (1 / |G_k|) sum_{i in G_k} sum_u 1/log2(1 + pos_u(i)) / |U|.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from project.metrics.accuracy import discounts


def group_recommendation_unfairness(values: pd.Series, groups: pd.Series, g1: str, g2: str) -> Dict[str, float]:
    """GRU between groups ``g1`` and ``g2`` of a per-user metric (users missing in ``groups`` are ignored)."""
    df = pd.DataFrame({"value": values, "group": groups.reindex(values.index)}).dropna()
    a, b = df.loc[df.group == g1, "value"], df.loc[df.group == g2, "value"]
    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"GRU needs users in both groups ({g1}: {len(a)}, {g2}: {len(b)})")
    return {"gru": abs(a.mean() - b.mean()), f"mean_{g1}": a.mean(), f"mean_{g2}": b.mean(),
            f"n_{g1}": int(len(a)), f"n_{g2}": int(len(b))}


def item_exposure(idx: np.ndarray, n_items: int, position_discount: bool = False) -> np.ndarray:
    """(I,) exposure of every catalogue item: number of lists containing it, or, with
    ``position_discount``, the sum of 1/log2(1 + rank) over those lists."""
    filled = idx >= 0
    weights = np.broadcast_to(discounts(idx.shape[1]) if position_discount else np.ones(idx.shape[1]), idx.shape)
    return np.bincount(idx[filled], weights=weights[filled], minlength=n_items).astype(float)


def gini_index(exposure: np.ndarray) -> float:
    e = np.sort(np.asarray(exposure, dtype=float))
    n, total = len(e), e.sum()
    if total == 0:
        return float("nan")
    j = np.arange(1, n + 1)
    return float(((2 * j - n - 1) * e).sum() / (n * total))


def shannon_entropy(exposure: np.ndarray, normalised: bool = True) -> float:
    e = np.asarray(exposure, dtype=float)
    total = e.sum()
    if total == 0 or (normalised and len(e) < 2):
        return float("nan")
    s = e[e > 0] / total
    h = float(-(s * np.log(s)).sum())
    return h / np.log(len(e)) if normalised else h


def group_exposure(idx: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """(n_groups,) mean position-discounted exposure per item and per list, for each item group."""
    groups = np.asarray(groups)
    e = item_exposure(idx, len(groups), position_discount=True) / len(idx)
    return np.array([e[groups == g].mean() if (groups == g).any() else np.nan for g in range(n_groups)])
