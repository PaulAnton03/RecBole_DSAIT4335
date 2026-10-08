"""Switching hybrid: route each user to a single model based on their group (Task 1.4).

For each user, the recommendation list comes entirely from one model, selected
by the user's membership in a group dimension (activity or taste).
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from project.metrics.evaluate import EvaluationContext
from project.utils.data_formats import (
    TOPK_CANDIDATES,
    load_recommendations,
)

DEFAULT_ASSIGNMENT_ACTIVITY = {"low": "ItemKNN", "medium": "EASE", "high": "NeuMF"}
DEFAULT_ASSIGNMENT_TASTE = {"niche": "ItemKNN", "mixed": "EASE", "mainstream": "NeuMF"}


def switching_recommendations(
    model_per_group: Dict[str, str],
    split: str,
    dimension: str = "activity",
    k: int = TOPK_CANDIDATES,
) -> pd.DataFrame:
    """Build a recommendation list by routing each user to a group-assigned model.

    Parameters
    ----------
    model_per_group : dict
        Maps group label (e.g. "low", "medium", "high") to a model name.
    split : "valid" or "test"
    dimension : "activity" or "taste"
    k : candidate list length (default 50)

    Returns
    -------
    pd.DataFrame with columns user_id, item_id, rank, score
    """
    ctx = EvaluationContext.for_split(split)
    user_groups = ctx.user_groups[dimension]

    needed_models = set(model_per_group.values())
    model_recs = {m: load_recommendations(m, split, k) for m in needed_models}

    parts = []
    for group_label, model_name in model_per_group.items():
        users_in_group = set(user_groups[user_groups == group_label].index)
        recs = model_recs[model_name]
        parts.append(recs[recs["user_id"].isin(users_in_group)])

    return pd.concat(parts, ignore_index=True).sort_values(
        ["user_id", "rank"]
    ).reset_index(drop=True)
