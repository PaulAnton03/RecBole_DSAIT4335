"""Hand-computed checks of our metrics (run: python -m pytest project/tests -q).

The accuracy cases are the lecture's worked example (W2S1 Evaluation Part 2, slides 9-14,
and W1S2 Evaluation Part 1, slide 36): four users, lists of four items.
"""
import numpy as np
import pandas as pd
import pytest

from project.metrics import EvaluationContext, accuracy, calibration, diversity, evaluate, fairness, lookups, novelty
from project.metrics import popularity

# ---------------------------------------------------------------- lecture example (K = 4)
LECTURE_GT = {"u1": {"i2", "i3"}, "u2": {"i1", "i2", "i3", "i5", "i7"}, "u3": {"i3", "i4", "i6"},
              "u4": {"i1", "i2", "i3", "i4", "i5", "i6"}}
LECTURE_LISTS = {"u1": ["i1", "i3", "i4", "i6"], "u2": ["i2", "i3", "i4", "i5"], "u3": ["i1", "i2", "i5", "i7"],
                 "u4": ["i3", "i4", "i5", "i6"]}
LECTURE_HITS = np.array([[0, 1, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=bool)
LECTURE_NREL = np.array([2, 5, 3, 6])


def test_lecture_precision_recall():
    np.testing.assert_allclose(accuracy.precision(LECTURE_HITS), [0.25, 0.75, 0.0, 1.0])
    assert accuracy.precision(LECTURE_HITS).mean() == pytest.approx(0.5)  # slide 36: "precision = 50%"
    np.testing.assert_allclose(accuracy.recall(LECTURE_HITS, LECTURE_NREL), [0.5, 0.6, 0.0, 4 / 6])


def test_lecture_ndcg():
    n = accuracy.ndcg(LECTURE_HITS, LECTURE_NREL)
    assert n[0] == pytest.approx(0.63 / 1.63, abs=2e-3)  # slide 9: 0.63 / 1.63 = 0.39
    assert n[1] == pytest.approx(2.06 / 2.56, abs=2e-3)  # slide 10: 2.06 / 2.56 = 0.8
    assert n[2] == 0.0 and n[3] == pytest.approx(1.0)  # u4: perfect ranking


def test_lecture_mrr_and_map_convention():
    assert accuracy.reciprocal_rank(LECTURE_HITS)[0] == pytest.approx(0.5)  # slide 12
    ap = accuracy.average_precision(LECTURE_HITS, LECTURE_NREL)
    # slide 14 sums 1 + 1 + 3/4 = 2.75 for u2 and divides by 3 (hits); we divide by min(K, |T|) = 4
    assert ap[1] == pytest.approx(2.75 / 4)
    assert ap[0] == pytest.approx(0.5 / 2) and ap[3] == pytest.approx(1.0)


def test_f1_and_hit():
    f = accuracy.f1(LECTURE_HITS, LECTURE_NREL)
    assert f[0] == pytest.approx(2 * 0.25 * 0.5 / 0.75) and f[2] == 0.0
    np.testing.assert_array_equal(accuracy.hit_rate(LECTURE_HITS), [1, 1, 0, 1])


def test_ndcg_short_relevant_set_is_normalised_by_min_k():
    hits = np.array([[True, False, False]])
    assert accuracy.ndcg(hits, np.array([1]))[0] == pytest.approx(1.0)  # IDCG uses min(K, |T|) = 1


def test_users_without_relevant_items_are_rejected():
    with pytest.raises(ValueError):
        accuracy.recall(LECTURE_HITS, np.array([2, 5, 0, 6]))


# -------------------------------------------------------------------------- diversity
GENRES = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=bool)  # A{x,y}, B{y}, C{z}, D{x,y}


def test_ild_identical_disjoint_and_mixed():
    ild = diversity.intra_list_diversity(np.array([[0, 3], [1, 2], [0, 1], [0, 2]]), GENRES)
    np.testing.assert_allclose(ild, [0.0, 1.0, 0.5, 1.0])
    # three items: d(A,B) = 1/2, d(A,C) = 1, d(B,C) = 1 -> mean over pairs 2.5 / 3 (not halved)
    assert diversity.intra_list_diversity(np.array([[0, 1, 2]]), GENRES)[0] == pytest.approx(2.5 / 3)


def test_ild_undefined_for_short_lists():
    ild = diversity.intra_list_diversity(np.array([[0, -1, -1], [0, 1, -1]]), GENRES)
    assert np.isnan(ild[0]) and ild[1] == pytest.approx(0.5)


def test_distance_matrix_matches_pairwise():
    d = diversity.jaccard_distance_matrix(GENRES)
    assert d[0, 1] == pytest.approx(0.5) and d[1, 2] == pytest.approx(1.0) and d[0, 3] == pytest.approx(0.0)


def test_coverage_counts_unique_items_globally():
    idx = np.array([[0, 1], [1, 2], [2, -1]])
    assert diversity.catalogue_coverage(idx, 5) == pytest.approx(3 / 5)


# ---------------------------------------------------------------------------- novelty
def test_novelty_is_self_information_and_decreases_with_popularity():
    counts = np.array([8, 2, 0])  # pop = 0.8, 0.2, unseen -> count 1 = 0.1
    info = novelty.self_information(counts)
    np.testing.assert_allclose(info, [-np.log2(0.8), -np.log2(0.2), -np.log2(0.1)])
    assert info[0] < info[1] < info[2]
    assert novelty.novelty(np.array([[0, 1], [2, -1]]), counts)[1] == pytest.approx(-np.log2(0.1))


def test_popularity_baseline_skips_history_and_serendipity_excludes_it():
    base = novelty.popularity_baseline_lists(np.array([5, 9, 1, 7]), [np.array([1])], k=2)
    np.testing.assert_array_equal(base, [[3, 0]])
    s = novelty.serendipity(np.array([[True, True, False]]), np.array([[0, 1, 2]]), np.array([[1, 5, 6]]))
    assert s[0] == pytest.approx(1 / 3)  # item 1 is relevant but expected (in the popularity list)


# ------------------------------------------------------------------------ calibration
def test_item_genre_distribution_is_normalised():
    np.testing.assert_allclose(calibration.item_genre_distribution(GENRES)[0], [0.5, 0.5, 0.0])


def test_miscalibration_zero_when_matching_and_finite_when_genre_missing():
    p = np.array([[0.5, 0.5, 0.0]])
    assert calibration.kl_miscalibration(p, p)[0] == pytest.approx(0.0)
    q = np.array([[1.0, 0.0, 0.0]])  # genre 2 of the history is absent from the list
    expected = 0.5 * np.log(0.5 / 0.995) + 0.5 * np.log(0.5 / 0.005)  # q~ = 0.99 q + 0.01 p
    assert calibration.kl_miscalibration(p, q)[0] == pytest.approx(expected)
    assert calibration.kl_miscalibration(q, p)[0] != pytest.approx(expected)  # direction KL(p || q~)


def test_miscalibration_from_lists_and_history():
    p_gi = calibration.item_genre_distribution(GENRES)
    history = np.array([[1, 0, 1, 0]], dtype=bool)  # A, C -> p = (1/4, 1/4, 1/2)
    np.testing.assert_allclose(calibration.profile(history, p_gi), [[0.25, 0.25, 0.5]])
    mc = calibration.miscalibration(np.array([[0, 2]]), history, p_gi)  # same genre mix as the history
    assert mc[0] == pytest.approx(0.0)


# ------------------------------------------------------------------- popularity groups
def test_popularity_groups_by_interaction_share():
    counts = np.array([50, 30, 10, 5, 3, 2, 0])  # 100 interactions
    g = lookups.popularity_groups(counts, 0.2, 0.2)
    # head: 50 alone holds >= 20 %; tail: items with <= 10 together hold exactly 20 %
    np.testing.assert_array_equal(g, [lookups.HEAD, lookups.MID] + [lookups.TAIL] * 5)


def test_upd_zero_for_same_mix_and_one_for_disjoint():
    groups = np.array([lookups.HEAD, lookups.HEAD, lookups.TAIL, lookups.TAIL])
    history = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
    upd = popularity.user_popularity_deviation(np.array([[0, 1], [2, 3]]), history, groups)
    np.testing.assert_allclose(upd, [0.0, 1.0])


def test_average_popularity_and_tail_share():
    counts, groups = np.array([4, 2, 0]), np.array([lookups.HEAD, lookups.MID, lookups.TAIL])
    np.testing.assert_allclose(popularity.average_popularity(np.array([[0, 1]]), counts, n_users=4), [0.75])
    np.testing.assert_allclose(popularity.group_share(np.array([[0, 2, -1]]), groups, lookups.TAIL), [0.5])


# --------------------------------------------------------------------------- fairness
def test_gru_with_group_sizes():
    values = pd.Series({"u1": 0.2, "u2": 0.4, "u3": 0.9})
    groups = pd.Series({"u1": "a", "u2": "a", "u3": "b"})
    out = fairness.group_recommendation_unfairness(values, groups, "a", "b")
    assert out["gru"] == pytest.approx(0.6) and out["n_a"] == 2 and out["n_b"] == 1
    equal = fairness.group_recommendation_unfairness(pd.Series({"u1": 0.3, "u3": 0.3}), groups, "a", "b")
    assert equal["gru"] == 0.0


def test_exposure_includes_unrecommended_items():
    e = fairness.item_exposure(np.array([[0, 1], [0, -1]]), n_items=4)
    np.testing.assert_array_equal(e, [2, 1, 0, 0])
    d = fairness.item_exposure(np.array([[0, 1]]), n_items=2, position_discount=True)
    np.testing.assert_allclose(d, [1.0, 1 / np.log2(3)])


def test_gini_and_entropy_extremes():
    assert fairness.gini_index(np.ones(5)) == pytest.approx(0.0)
    assert fairness.gini_index(np.array([0, 0, 0, 7])) == pytest.approx(3 / 4)  # (n - 1) / n
    assert fairness.gini_index(np.array([3, 1])) == pytest.approx(0.25)
    assert fairness.shannon_entropy(np.ones(5)) == pytest.approx(1.0)
    assert fairness.shannon_entropy(np.array([2, 2, 0, 0])) == pytest.approx(0.5)  # ln 2 / ln 4


def test_global_metrics_do_not_depend_on_batches():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 30, size=(40, 5))
    full = fairness.item_exposure(idx, 30)
    halves = fairness.item_exposure(idx[:17], 30) + fairness.item_exposure(idx[17:], 30)
    np.testing.assert_array_equal(full, halves)
    assert fairness.gini_index(full) == pytest.approx(fairness.gini_index(halves))


# ------------------------------------------------------- cross-check with RecBole's code
def test_global_metrics_match_recbole_implementation():
    rb = pytest.importorskip("recbole.evaluator.metrics")
    rng = np.random.default_rng(1)
    n_items, idx = 60, np.stack([rng.choice(40, size=10, replace=False) for _ in range(25)])
    exposure = fairness.item_exposure(idx, n_items)
    gini = rb.GiniIndex.__new__(rb.GiniIndex)
    assert fairness.gini_index(exposure) == pytest.approx(gini.get_gini(idx, n_items))
    cov = rb.ItemCoverage.__new__(rb.ItemCoverage)
    assert diversity.catalogue_coverage(idx, n_items) == pytest.approx(cov.get_coverage(idx, n_items))
    ent = rb.ShannonEntropy.__new__(rb.ShannonEntropy)  # RecBole divides by the number of distinct items
    assert fairness.shannon_entropy(exposure, normalised=False) == pytest.approx(
        ent.get_entropy(idx) * len(np.unique(idx)))
    counts = rng.integers(0, 50, size=n_items)
    pop = rb.AveragePopularity.__new__(rb.AveragePopularity)
    recbole_avg = pop.metric_info(pop.get_pop(idx, dict(enumerate(counts))).astype(float))[:, -1]
    np.testing.assert_allclose(popularity.average_popularity(idx, counts, n_users=1), recbole_avg)


# ---------------------------------------------------------------- end-to-end evaluate()
def _lecture_context():
    items = [f"i{j}" for j in range(1, 9)]
    item_genres = {i: (["Drama"] if j % 2 else ["Comedy", "Drama"]) for j, i in enumerate(items)}
    history = {"u1": {"i5", "i7"}, "u2": {"i6"}, "u3": {"i8"}, "u4": {"i7", "i8"}}
    return EvaluationContext.build(LECTURE_GT, history, item_genres, items)


def _frame(lists):
    return pd.DataFrame([(u, i, r, 1.0 / r) for u, items in lists.items() for r, i in enumerate(items, 1)],
                        columns=["user_id", "item_id", "rank", "score"])


def test_evaluate_reproduces_lecture_example():
    res = evaluate(_frame(LECTURE_LISTS), _lecture_context(), k=4)
    pu = res.per_user.loc[["u1", "u2", "u3", "u4"]]
    np.testing.assert_allclose(pu["precision@4"], accuracy.precision(LECTURE_HITS))
    np.testing.assert_allclose(pu["ndcg@4"], accuracy.ndcg(LECTURE_HITS, LECTURE_NREL))
    assert res.summary["precision@4"] == pytest.approx(0.5)
    assert res.summary["coverage@4"] == pytest.approx(7 / 8)  # i8 is never recommended
    assert res.summary["users"] == 4 and res.summary["users_missing_list"] == 0
    assert res.item_exposure["i8"] == 0 and res.item_exposure["i3"] == 3  # in the lists of u1, u2, u4


def test_evaluate_missing_list_scores_zero_and_extra_users_are_ignored():
    lists = {u: items for u, items in LECTURE_LISTS.items() if u != "u4"}
    lists["stranger"] = ["i1"]
    res = evaluate(_frame(lists), _lecture_context(), k=4)
    assert res.per_user.loc["u4", "ndcg@4"] == 0.0 and np.isnan(res.per_user.loc["u4", "ild@4"])
    assert res.summary["users_missing_list"] == 1 and res.summary["users_ignored"] == 1
    assert res.summary["ndcg@4"] == pytest.approx(res.per_user["ndcg@4"].mean())


def test_evaluate_rejects_broken_lists():
    ctx = _lecture_context()
    dup = _frame({"u1": ["i1", "i1", "i3", "i4"]})
    with pytest.raises(ValueError, match="same item"):
        evaluate(dup, ctx, k=4)
    with pytest.raises(ValueError, match="history"):
        evaluate(_frame({"u1": ["i5", "i3"]}), ctx, k=4)  # i5 is in u1's history
    with pytest.raises(ValueError, match="catalogue"):
        evaluate(_frame({"u1": ["i99"]}), ctx, k=4)


def test_evaluate_uses_only_top_k_ranks():
    res = evaluate(_frame({u: items + ["i8"] for u, items in LECTURE_LISTS.items() if u != "u3"}),
                   _lecture_context(), k=4)
    assert res.item_exposure["i8"] == 0  # rank 5 is outside the final list
