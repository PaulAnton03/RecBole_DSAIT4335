"""User- and item-side fairness (W3S2 Fairness, slides 38, 42, 43).

User side -- Group Recommendation Unfairness (slide 38, Fu et al. SIGIR 2020):

    GRU(G1, G2) = | mean_{u in G1} F(u) - mean_{u in G2} F(u) |

with F a per-user quality metric (we use NDCG@K, the slide's example). For more than two groups
we use the max-min extension, max_g mean_g F - min_g mean_g F (equal to the slide's GRU for two
groups). Lower = smaller quality gap; always reported with every group's mean and size, since a
small gap alone does not mean good quality (it can come from lowering the best-served group).

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

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from project.metrics.accuracy import discounts


def group_recommendation_unfairness(values: pd.Series, groups: pd.Series,
                                    group_names: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """GRU (max - min of the group means) of a per-user metric, with every group's mean and size.

    ``group_names`` fixes which groups are reported (default: all groups found); empty groups get a
    NaN mean and are left out of the gap, which is NaN with fewer than two non-empty groups. Users
    missing in ``groups`` are ignored.
    """
    df = pd.DataFrame({"value": values, "group": groups.reindex(values.index)}).dropna()
    names = list(group_names) if group_names is not None else sorted(df["group"].unique())
    means = {g: df.loc[df.group == g, "value"].mean() for g in names}  # NaN for an empty group
    sizes = {g: int((df.group == g).sum()) for g in names}
    present = [means[g] for g in names if sizes[g] > 0]
    gru = max(present) - min(present) if len(present) >= 2 else float("nan")  # undefined, not 0
    return {"gru": gru, "means": means, "sizes": sizes}


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
