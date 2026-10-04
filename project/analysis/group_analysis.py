"""User- and item-group analysis of any recommendation list (Task 2.5; reused for Task 3.4).

Groups are frozen and built from the training split only (``project.metrics.lookups``):

* users, two separate dimensions (tertiles of a per-user score, ties kept together):
  activity -- number of training interactions (low / medium / high);
  taste    -- share of the user's training interactions on head items (niche / mixed / mainstream);
* items: head / mid / tail by cumulative training-interaction mass (20 / 60 / 20 %), plus unseen
  (no training interaction).

Per-user metrics are computed on each user's complete list (``project.metrics.evaluate``) and then
averaged within a group, with a 95 % percentile-bootstrap interval over the group's users. The
max-min gap across a dimension's groups is the GRU extension of the fairness lecture; it is always
reported next to the group means and sizes.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from project.metrics import EvaluationContext, evaluate, item_groups, lookups, popularity
from project.metrics.evaluate import index_matrix
from project.utils.data_formats import TOPK_FINAL

DIMENSIONS = {"activity": lookups.ACTIVITY_GROUPS, "taste": lookups.TASTE_GROUPS}
USER_METRICS = ["ndcg", "recall", "ild", "novelty", "miscalibration", "upd", "tailshare"]
N_BOOT, SEED = 1000, 2020


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05):
    """Percentile-bootstrap interval of the mean (NaNs dropped); (nan, nan) for fewer than 2 values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, np.nan
    means = v[np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def user_group_metrics(per_user: pd.DataFrame, name: str, k: int = TOPK_FINAL, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Long table: name, dimension, group, users, metric, mean, ci_low, ci_high."""
    rows = []
    for dim, groups in DIMENSIONS.items():
        for g in groups:
            sub = per_user[per_user[dim] == g]
            for m in USER_METRICS:
                col = f"{m}@{k}"
                lo, hi = bootstrap_ci(sub[col].to_numpy(), n_boot)
                rows.append({"name": name, "dimension": dim, "group": g, "users": len(sub), "metric": col,
                             "mean": sub[col].mean(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def group_gaps(user_df: pd.DataFrame) -> pd.DataFrame:
    """Max-min gap of every metric across the groups of a dimension (GRU for NDCG), with the groups involved."""
    rows = []
    for (name, dim, metric), sub in user_df.groupby(["name", "dimension", "metric"], sort=False):
        sub = sub.dropna(subset=["mean"])
        best, worst = sub.loc[sub["mean"].idxmax()], sub.loc[sub["mean"].idxmin()]
        rows.append({"name": name, "dimension": dim, "metric": metric, "gap": best["mean"] - worst["mean"],
                     "highest_group": best["group"], "highest": best["mean"],
                     "lowest_group": worst["group"], "lowest": worst["mean"]})
    return pd.DataFrame(rows)


def item_group_metrics(idx: np.ndarray, ctx: EvaluationContext, name: str, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Item-group accuracy and exposure (``project.metrics.item_groups``) with a user-bootstrap
    interval for the micro recall of each group."""
    names = lookups.POPULARITY_GROUPS
    table = item_groups.item_group_table(idx, ctx.relevant, ctx.item_groups, names)
    hits, support, _ = item_groups.per_user_group_counts(idx, ctx.relevant, ctx.item_groups, len(names))
    sample = np.random.default_rng(SEED).integers(0, len(idx), size=(n_boot, len(idx)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = hits[sample].sum(axis=1) / support[sample].sum(axis=1)  # (n_boot, n_groups)
    table["recall_ci_low"] = np.nanquantile(boot, 0.025, axis=0) if len(boot) else np.nan
    table["recall_ci_high"] = np.nanquantile(boot, 0.975, axis=0) if len(boot) else np.nan
    table.loc[table["support"] == 0, ["recall_ci_low", "recall_ci_high"]] = np.nan
    return table.assign(name=name)[["name"] + [c for c in table.columns]]


def popularity_alignment(idx: np.ndarray, ctx: EvaluationContext, name: str) -> pd.DataFrame:
    """Mean share of head / mid / tail / unseen items in the users' training profiles and in their
    lists, per taste group: does a model follow niche users' taste, or only show more tail items?"""
    n = len(lookups.POPULARITY_GROUPS)
    profile = popularity.group_distribution_of_histories(ctx.train, ctx.item_groups, n)
    lists = popularity.group_distribution_of_lists(idx, ctx.item_groups, n)
    taste = ctx.user_groups.reindex(ctx.users)["taste"].to_numpy()
    rows = []
    for g in lookups.TASTE_GROUPS:
        mask = taste == g
        for source, dist in (("profile", profile), ("list", lists)):
            shares = np.nanmean(dist[mask], axis=0) if mask.any() else np.full(n, np.nan)  # empty group: NaN
            rows.append({"name": name, "taste": g, "source": source, "users": int(mask.sum()),
                         **dict(zip(lookups.POPULARITY_GROUPS, shares))})
    return pd.DataFrame(rows)


def intersection(per_user: pd.DataFrame, name: str, k: int = TOPK_FINAL) -> pd.DataFrame:
    """Activity x taste cells: users and mean NDCG / recall (optional deeper look; small cells are unstable)."""
    out = (per_user.groupby(["activity", "taste"])[[f"ndcg@{k}", f"recall@{k}"]].agg(["size", "mean"]))
    out.columns = ["users", f"ndcg@{k}", "_", f"recall@{k}"]
    return out.drop(columns="_").reset_index().assign(name=name)


def paired_differences(per_user: pd.DataFrame, names: Sequence[str], k: int = TOPK_FINAL,
                       metrics: Sequence[str] = ("ndcg", "recall"), n_boot: int = N_BOOT) -> pd.DataFrame:
    """Within-group comparison of two lists on the same users: mean per-user difference A - B with a
    paired 95 % bootstrap interval, for every pair of ``names``, every group and all users.

    ``per_user`` is the long per-user table of ``analyse`` (columns name, user_id, activity, taste, metrics).
    Comparing models inside a group avoids the held-out-set-size confound of raw group gaps.
    """
    wide = {m: per_user.pivot(index="user_id", columns="name", values=f"{m}@{k}") for m in metrics}
    groups = per_user.drop_duplicates("user_id").set_index("user_id")[list(DIMENSIONS)]
    cells = [("all", "all", np.ones(len(groups), dtype=bool))] + [
        (dim, g, (groups[dim] == g).to_numpy()) for dim, gs in DIMENSIONS.items() for g in gs]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for m in metrics:
                d = (wide[m][a] - wide[m][b]).reindex(groups.index).to_numpy()
                for dim, g, mask in cells:
                    lo, hi = bootstrap_ci(d[mask], n_boot)
                    rows.append({"a": a, "b": b, "metric": f"{m}@{k}", "dimension": dim, "group": g,
                                 "users": int(mask.sum()), "diff": np.nanmean(d[mask]), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def analyse_list(recs: pd.DataFrame, ctx: EvaluationContext, name: str, k: int = TOPK_FINAL,
                 n_boot: int = N_BOOT) -> Dict[str, pd.DataFrame]:
    """Every group table for one list."""
    res = evaluate(recs, ctx, k, name)
    idx, _ = index_matrix(recs, ctx, k)
    users = user_group_metrics(res.per_user, name, k, n_boot)
    return {"user": users, "gaps": group_gaps(users), "item": item_group_metrics(idx, ctx, name, n_boot),
            "alignment": popularity_alignment(idx, ctx, name), "intersection": intersection(res.per_user, name, k),
            "per_user": res.per_user.reset_index().assign(name=name)}


def analyse(lists: Dict[str, pd.DataFrame], ctx: EvaluationContext, k: int = TOPK_FINAL,
            n_boot: int = N_BOOT) -> Dict[str, pd.DataFrame]:
    """``analyse_list`` for several lists, concatenated per table (one ``name`` column), plus the
    paired within-group differences between every two lists (``pairs``)."""
    parts = [analyse_list(recs, ctx, name, k, n_boot) for name, recs in lists.items()]
    out = {key: pd.concat([p[key] for p in parts], ignore_index=True) for key in parts[0]}
    out["pairs"] = paired_differences(out["per_user"], list(lists), k, n_boot=n_boot)
    return out


def model_order(names: Sequence[str], registry_order: Sequence[str]) -> list:
    order = {m: i for i, m in enumerate(registry_order)}
    return sorted(names, key=lambda n: (order.get(n, len(order)), n))
