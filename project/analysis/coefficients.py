"""Coefficient analysis of the weighted hybrid (Task 2.3; Track C): what each member contributes.

The learned weights alone mislead when two members are similar (they split the credit), so
four views are combined:

* bootstrap intervals of the weights (resample users, refit);
* leave-one-member-out: refit without the member and measure the drop in NDCG@10 -- the
  member's real contribution (computed in ``project.experiments.hybrid``, which owns the data);
* hit attribution: for every correct movie in the hybrid's top-10, the member whose weighted
  term w_k (z_k - mean of the member's scores for that user) is largest -- the member that singles
  the movie out most strongly relative to how it scores the user's other candidates -- plus the
  softer mean share of the positive weighted terms, and in how many members' own top-10 lists the
  movie appeared (0 = the hybrid found it although no member ranked it in its top-10);
* member overlap: correlation of the members' rescaled scores on the candidate rows and the
  mean Jaccard overlap of their top-10 lists -- why a strong model can get a small weight.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from project.hybrids.weighted import SEED, MemberScores, RegressionTable, Weights, fit_weights, lists_from
from project.metrics import EvaluationContext, lookups
from project.metrics.evaluate import index_matrix
from project.utils.data_formats import TOPK_FINAL


def bootstrap_weights(table: RegressionTable, members: Sequence[str], regression: str, lam: float,
                      n_boot: int = 200, seed: int = SEED) -> pd.DataFrame:
    """95 % percentile-bootstrap interval of every standardised weight: users are resampled with
    replacement (all rows of a drawn user are kept together) and the regression is refitted."""
    n_users = int(table.user.max()) + 1
    counts = np.bincount(table.user, minlength=n_users)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.integers(0, n_users, n_users)
        rows = np.concatenate([np.arange(starts[u], starts[u] + counts[u]) for u in sample])
        boots.append(fit_weights(table.rows(rows), members, regression, lam).coef_std)
    boots = np.array(boots)
    return pd.DataFrame({"member": list(members), "ci_low": np.quantile(boots, 0.025, axis=0),
                         "ci_high": np.quantile(boots, 0.975, axis=0), "boot_sd": boots.std(axis=0)})


def hit_attribution(ms_test: MemberScores, w: Weights, hyb_test: np.ndarray, ctx: EvaluationContext,
                    member_top: Sequence[np.ndarray], k: int = TOPK_FINAL):
    """Attribute every hit of the hybrid's top-k test list to the member with the largest weighted
    term. Returns (per-member table, hits-by-number-of-members table)."""
    idx, _ = index_matrix(lists_from(hyb_test, ctx, k), ctx, k)
    filled = idx >= 0
    rows = np.broadcast_to(np.arange(len(idx))[:, None], idx.shape)
    hits = ctx.relevant[rows, np.where(filled, idx, 0)] & filled
    hu, hp = np.nonzero(hits)
    hi = idx[hu, hp]
    centre = np.nanmean(ms_test.z, axis=2)  # (K, U): each member's mean score over the user's candidates
    contrib = w.coef[:, None] * (ms_test.z[:, hu, hi] - centre[:, hu])  # (K, n_hits)
    attributed = contrib.argmax(axis=0)
    positive = np.clip(contrib, 0.0, None)  # soft version: each member's share of the positive contributions
    contrib_share = positive / np.where(positive.sum(axis=0) > 0, positive.sum(axis=0), 1.0)
    in_own = np.stack([(member_top[j][hu, :k] == hi[:, None]).any(axis=1) for j in range(len(ms_test.members))])
    n_in_own = in_own.sum(axis=0)
    group = ctx.item_groups[hi]
    names = lookups.POPULARITY_GROUPS
    per_member = []
    for j, m in enumerate(ms_test.members):
        mine = attributed == j
        own = member_top[j][:, :k]
        own_filled = own >= 0
        own_hits = int((ctx.relevant[np.broadcast_to(np.arange(len(own))[:, None], own.shape), np.where(own_filled, own, 0)]
                        & own_filled).sum())
        per_member.append({"member": m, "attributed_hits": int(mine.sum()), "attributed_share": float(mine.mean()),
                           "mean_contribution_share": float(contrib_share[j].mean()),
                           **{f"attributed_{g}": int((mine & (group == c)).sum()) for c, g in enumerate(names)},
                           "hybrid_hits_in_own_top10": int(in_own[j].sum()), "own_top10_hits": own_hits})
    by_count = pd.DataFrame({"members_with_hit_in_top10": np.arange(len(ms_test.members) + 1),
                             "hits": np.bincount(n_in_own, minlength=len(ms_test.members) + 1)})
    by_count["share"] = by_count["hits"] / max(len(hu), 1)
    return pd.DataFrame(per_member), by_count


def member_overlap(ms: MemberScores, cand: np.ndarray, member_top: Sequence[np.ndarray], k: int = TOPK_FINAL) -> pd.DataFrame:
    """Pairwise Pearson correlation of the rescaled scores on the candidate rows and mean Jaccard
    overlap of the members' top-k lists (long format, one row per unordered pair)."""
    X = ms.z[:, cand]
    corr = np.corrcoef(X)
    rows = []
    for a in range(len(ms.members)):
        for b in range(a + 1, len(ms.members)):
            la, lb = member_top[a][:, :k], member_top[b][:, :k]
            inter = (la[:, :, None] == lb[:, None, :]).any(axis=2).sum(axis=1)
            union = (la >= 0).sum(axis=1) + (lb >= 0).sum(axis=1) - inter
            jac = np.divide(inter, union, out=np.full(len(la), np.nan), where=union > 0)
            rows.append({"member_a": ms.members[a], "member_b": ms.members[b], "score_correlation": float(corr[a, b]),
                         f"top{k}_jaccard": float(np.nanmean(jac))})
    return pd.DataFrame(rows)
