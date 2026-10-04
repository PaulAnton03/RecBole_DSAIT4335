"""Item-group metrics (Task 2.5): accuracy and exposure of the head / mid / tail / unseen groups.

Two different questions are asked for every item group G, on the complete original lists (ranks
are kept; no list is filtered to one group and re-ranked):

* accuracy -- are the relevant items of G retrieved?

      recall_G = sum_u |L_u ∩ T_u ∩ G| / sum_u |T_u ∩ G|     (micro: every held-out pair counts once)

  reported with its support sum_u |T_u ∩ G| (held-out pairs of the group), and hit_share_G, the
  share of all hits that fall in G.

* exposure -- how much recommendation attention does G receive?

      slot_share_G       share of all recommendation slots holding an item of G
      slot_share_disc_G  the same with slots weighted 1/log2(1 + rank)
      coverage_G         share of G's items recommended at least once (within-group coverage)
      lists_per_item_G   slots of G / |G|: in how many lists an average item of G appears

ILD and calibration are properties of whole lists and have no per-item-group version.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from project.metrics.accuracy import discounts


def per_user_group_counts(idx: np.ndarray, relevant: np.ndarray, item_groups: np.ndarray, n_groups: int):
    """Per user and group: hits |L ∩ T ∩ G| (U, n_groups), support |T ∩ G| (U, n_groups), slots |L ∩ G|."""
    filled = idx >= 0
    safe = np.where(filled, idx, 0)
    hits = relevant[np.arange(len(idx))[:, None], safe] & filled
    slot_group = np.where(filled, np.asarray(item_groups)[safe], -1)
    one_hot = np.eye(n_groups, dtype=int)[np.asarray(item_groups)]  # (I, n_groups)
    support = relevant.astype(int) @ one_hot
    hit_counts = np.stack([(hits & (slot_group == g)).sum(axis=1) for g in range(n_groups)], axis=1)
    slots = np.stack([(slot_group == g).sum(axis=1) for g in range(n_groups)], axis=1)
    return hit_counts, support, slots


def item_group_table(idx: np.ndarray, relevant: np.ndarray, item_groups: np.ndarray,
                     group_names: Sequence[str]) -> pd.DataFrame:
    """One row per item group with the accuracy and exposure measures above."""
    n_groups = len(group_names)
    item_groups = np.asarray(item_groups)
    hits, support, slots = per_user_group_counts(idx, relevant, item_groups, n_groups)
    filled = idx >= 0
    safe = np.where(filled, idx, 0)
    weights = np.where(filled, discounts(idx.shape[1]), 0.0)
    slot_group = np.where(filled, item_groups[safe], -1)
    recommended = np.zeros(len(item_groups), dtype=bool)
    recommended[idx[filled]] = True
    rows = []
    for g, name in enumerate(group_names):
        size = int((item_groups == g).sum())
        sup, hit = int(support[:, g].sum()), int(hits[:, g].sum())
        rows.append({
            "group": name,
            "items": size,
            "support": sup,
            "hits": hit,
            "recall": hit / sup if sup else np.nan,
            "hit_share": hit / hits.sum() if hits.sum() else np.nan,
            "slot_share": slots[:, g].sum() / filled.sum(),
            "slot_share_disc": weights[slot_group == g].sum() / weights.sum(),
            "coverage": recommended[item_groups == g].mean() if size else np.nan,
            "lists_per_item": slots[:, g].sum() / size if size else np.nan,
        })
    return pd.DataFrame(rows)
