"""Hand-computed checks of the item-group metrics and the group analysis (Task 2.5)."""
import numpy as np
import pandas as pd
import pytest

from project.analysis import group_analysis as ga
from project.metrics import EvaluationContext, item_groups, lookups

H, M, T, U = lookups.HEAD, lookups.MID, lookups.TAIL, lookups.UNSEEN
GROUPS = np.array([H, H, M, T, U])  # items 0..4
NAMES = lookups.POPULARITY_GROUPS


def test_item_group_table_by_hand():
    idx = np.array([[0, 2, 3], [1, 0, -1]])  # user 0: head, mid, tail; user 1: head, head, empty slot
    relevant = np.zeros((2, 5), dtype=bool)
    relevant[0, [0, 3, 4]] = True  # user 0: one head hit, one tail hit, unseen item 4 missed
    relevant[1, [2]] = True  # user 1: a mid item that is not recommended
    tab = item_groups.item_group_table(idx, relevant, GROUPS, NAMES).set_index("group")
    assert tab.loc["head", "recall"] == 1.0 and tab.loc["mid", "recall"] == 0.0
    assert tab.loc["tail", "recall"] == 1.0 and tab.loc["unseen", "recall"] == 0.0
    assert tab.loc["mid", "support"] == 1 and tab.loc["unseen", "support"] == 1
    assert tab.loc["head", "hit_share"] == 0.5  # 2 hits in total: head and tail
    assert tab.loc["head", "slot_share"] == pytest.approx(3 / 5)  # 5 filled slots, 3 on head items
    assert tab.loc["head", "coverage"] == 1.0 and tab.loc["unseen", "coverage"] == 0.0
    assert tab.loc["head", "lists_per_item"] == pytest.approx(3 / 2)
    d = 1 / np.log2([2, 3, 4])
    assert tab.loc["tail", "slot_share_disc"] == pytest.approx(d[2] / (d.sum() + d[:2].sum()))


def test_item_group_recall_keeps_original_ranks():
    idx = np.array([[2, 0]])  # the head item sits at rank 2 and stays there
    relevant = np.zeros((1, 5), dtype=bool)
    relevant[0, 0] = True
    tab = item_groups.item_group_table(idx, relevant, GROUPS, NAMES).set_index("group")
    assert tab.loc["head", "slot_share_disc"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def test_bootstrap_ci_brackets_the_mean_and_needs_two_values():
    v = np.arange(100, dtype=float)
    lo, hi = ga.bootstrap_ci(v, n_boot=500)
    assert lo < v.mean() < hi
    assert all(np.isnan(ga.bootstrap_ci(np.array([1.0]))))


def _per_user():
    users = [f"u{i}" for i in range(6)]
    activity = ["low", "low", "medium", "medium", "high", "high"]
    taste = ["niche", "mixed", "mainstream", "niche", "mixed", "mainstream"]
    rows = []
    for name, base in (("A", 0.5), ("B", 0.2)):
        for i, u in enumerate(users):
            rows.append({"name": name, "user_id": u, "activity": activity[i], "taste": taste[i],
                         "ndcg@10": base + 0.1 * i, "recall@10": base})
    return pd.DataFrame(rows)


def test_paired_differences_and_gaps():
    pu = _per_user()
    pairs = ga.paired_differences(pu, ["A", "B"], k=10, n_boot=200)
    row = pairs[(pairs.metric == "ndcg@10") & (pairs.group == "all")].iloc[0]
    assert row["diff"] == pytest.approx(0.3) and row["ci_low"] == pytest.approx(0.3)  # constant difference
    assert set(pairs.group) == {"all", "low", "medium", "high", "niche", "mixed", "mainstream"}
    users = ga.user_group_metrics(pu[pu.name == "A"].set_index("user_id").assign(
        **{f"{m}@10": 0.0 for m in ga.USER_METRICS if m not in ("ndcg", "recall")}), "A", k=10, n_boot=50)
    gaps = ga.group_gaps(users).set_index(["dimension", "metric"])
    # NDCG of A by activity: low 0.55, medium 0.75, high 0.95 -> gap 0.4 (max - min)
    assert gaps.loc[("activity", "ndcg@10"), "gap"] == pytest.approx(0.4)
    assert gaps.loc[("activity", "ndcg@10"), "lowest_group"] == "low"


def test_popularity_alignment_shares_sum_to_one():
    items = [f"i{j}" for j in range(6)]
    genres = {i: ["Drama"] for i in items}
    history = {"u1": {"i0", "i1"}, "u2": {"i0", "i2"}, "u3": {"i3", "i4"}}
    truth = {"u1": {"i5"}, "u2": {"i5"}, "u3": {"i5"}}
    ctx = EvaluationContext.build(truth, history, genres, items)
    idx = np.array([[2, 3], [1, 3], [0, 1]])
    out = ga.popularity_alignment(idx, ctx, "X")
    filled = out[out.users > 0]
    np.testing.assert_allclose(filled[list(lookups.POPULARITY_GROUPS)].sum(axis=1), 1.0)
    assert out[out.users == 0][list(lookups.POPULARITY_GROUPS)].isna().all().all()  # empty group: undefined
