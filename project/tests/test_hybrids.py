"""Hand-computed checks of the weighted hybrid (Track C; run: python -m pytest project/tests -q)."""
import numpy as np
import pandas as pd
import pytest

from project.analysis.coefficients import bootstrap_weights, hit_attribution, member_overlap
from project.hybrids import weighted as wh
from project.hybrids.weighted_tuning import HybridData, Setting, cv_score, default_grid, fit_hybrid
from project.metrics import EvaluationContext
from project.utils import data_formats as df_

ITEMS = [f"i{j}" for j in range(8)]
HISTORY = {"u1": {"i0"}, "u2": {"i1"}, "u3": {"i2"}, "u4": set()}
TRUTH = {"u1": {"i3", "i4"}, "u2": {"i5"}, "u3": {"i6", "i7"}, "u4": {"i0"}}


def _ctx(split="valid"):
    return EvaluationContext.build(TRUTH, HISTORY, {i: ["Drama"] for i in ITEMS}, ITEMS, split=split)


def _raw(ctx):
    """Member A ranks every relevant item first; member B is constant (uninformative)."""
    a = np.tile(np.linspace(0.3, 0.1, ctx.n_items, dtype=np.float32), (len(ctx.users), 1))
    a[ctx.relevant] = 0.9
    a[ctx.history] = np.nan
    b = np.where(ctx.history, np.nan, 0.5).astype(np.float32)
    return {"A": a, "B": b}


def _data(ctx, raw=None):
    raw = raw or _raw(ctx)
    lists = {m: wh.member_top_lists(m, "valid", ctx, 50, raw[m]) for m in raw}  # no list files -> from the scores
    return HybridData(ctx, raw, lists, ctx, raw)


# ----------------------------------------------------------------------------------------- rescale
def test_rescale_methods_keep_nan_and_rescale_per_user():
    m = np.array([[1.0, 3.0, np.nan, 5.0], [2.0, 2.0, 2.0, np.nan]], dtype=np.float32)
    z = wh.rescale(m, "zscore")
    assert np.isnan(z[0, 2]) and np.isnan(z[1, 3])
    assert np.nanmean(z[0]) == pytest.approx(0.0, abs=1e-6) and np.nanstd(z[0]) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(z[1, :3], 0.0)  # constant row -> 0, not NaN
    mm = wh.rescale(m, "minmax")
    np.testing.assert_allclose(mm[0, [0, 1, 3]], [0.0, 0.5, 1.0])
    r = wh.rescale(m, "rank")
    np.testing.assert_allclose(r[0, [0, 1, 3]], [1 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(r[1, :3], 2 / 3)  # ties averaged
    assert np.array_equal(np.isnan(wh.rescale(m, "none")), np.isnan(m))
    with pytest.raises(ValueError):
        wh.rescale(m, "sigmoid")


def test_aligned_scores_reorders_and_checks_the_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(df_, "SCORES_DIR", tmp_path)
    ctx = _ctx()
    raw = _raw(ctx)["A"]
    order_u, order_i = [3, 0, 2, 1], list(range(7, -1, -1))  # shuffled on disk
    df_.save_scores(df_.ScoreMatrix(ctx.users[order_u], ctx.catalogue[order_i], raw[order_u][:, order_i]), "A", "valid")
    back = wh.aligned_scores("A", "valid", ctx)
    np.testing.assert_array_equal(np.isnan(back), np.isnan(raw))
    np.testing.assert_allclose(np.nan_to_num(back), np.nan_to_num(raw))
    wrong = raw.copy()
    wrong[0, 5] = np.nan  # masks a candidate that is not history
    df_.save_scores(df_.ScoreMatrix(ctx.users, ctx.catalogue, wrong), "A", "valid")
    with pytest.raises(ValueError, match="masked history"):
        wh.aligned_scores("A", "valid", ctx)


def test_aligned_scores_restores_blanked_top_scores_from_the_list(tmp_path, monkeypatch):
    """Track A's release stores each user's top-10 scores as NaN; they come back from the list."""
    monkeypatch.setattr(df_, "SCORES_DIR", tmp_path)
    monkeypatch.setattr(df_, "RECOMMENDATIONS_DIR", tmp_path)
    ctx = _ctx()
    raw = _raw(ctx)["A"]
    recs = df_.topk_from_scores(df_.ScoreMatrix(ctx.users, ctx.catalogue, raw), 50)
    df_.save_recommendations(recs, "A", "valid", 50)
    blanked = raw.copy()
    top = recs[recs["rank"] <= 2]  # pretend the export blanked the two best scores of every user
    blanked[pd.Index(ctx.users).get_indexer(top.user_id), pd.Index(ctx.catalogue).get_indexer(top.item_id)] = np.nan
    df_.save_scores(df_.ScoreMatrix(ctx.users, ctx.catalogue, blanked), "A", "valid")
    back = wh.aligned_scores("A", "valid", ctx)
    np.testing.assert_array_equal(np.isnan(back), ctx.history)
    np.testing.assert_allclose(np.nan_to_num(back), np.nan_to_num(raw), rtol=1e-5)


# -------------------------------------------------------------------------------------- candidates
def test_candidate_masks():
    ctx = _ctx()
    assert np.array_equal(wh.candidate_mask(ctx, "all"), ~ctx.history)
    top = [np.array([[3, -1], [5, 6], [6, 7], [0, 1]]), np.array([[4, -1], [5, -1], [7, -1], [2, -1]])]
    union = wh.candidate_mask(ctx, "union", top)
    assert union.sum() == 9 and union[0, 3] and union[0, 4] and not union[0, 5]
    with pytest.raises(ValueError):
        wh.candidate_mask(ctx, "union", [np.array([[0, -1], [1, -1], [2, -1], [3, -1]])])  # i0 is u1's history


# ---------------------------------------------------------------------------------------- fitting
def test_ridge_recovers_a_known_linear_combination():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 3))
    y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + 0.0 * X[:, 2] + 0.05 * rng.normal(size=5000)
    w = wh.fit_weights(wh.RegressionTable.build(X, y, np.zeros(5000, dtype=int)), ["a", "b", "c"], "ridge", lam=0.0)
    np.testing.assert_allclose(w.coef, [2.0, 1.0, 0.0], atol=0.02)
    np.testing.assert_allclose(w.coef_std, w.coef * X.std(axis=0), atol=1e-6)
    assert w.share.sum() == pytest.approx(1.0)


def test_nnls_and_logistic_signs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4000, 2))
    y = (X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.normal(size=4000) > 0).astype(float)
    nn = wh.fit_weights(wh.RegressionTable.build(X, y, np.zeros(4000, dtype=int)), ["a", "b"], "nnls", lam=0.0)
    assert nn.coef[0] > 0 and nn.coef[1] == 0.0  # the negative direction is clipped at zero
    lg = wh.fit_weights(wh.RegressionTable.build(X, y, np.zeros(4000, dtype=int)), ["a", "b"], "logistic", lam=1e-3)
    assert lg.coef[0] > 0 > lg.coef[1] and abs(lg.coef[0] / lg.coef[1]) == pytest.approx(2.0, rel=0.25)


def test_single_member_weight_reproduces_that_member():
    ctx = _ctx()
    raw = _raw(ctx)
    ms = wh.MemberScores.from_raw(["A", "B"], "valid", raw, "zscore")
    w = wh.Weights(["A", "B"], np.array([1.0, 0.0]), np.array([1.0, 0.0]), "ridge", 0.0)
    hyb = wh.combine(ms, w)
    assert np.array_equal(np.isnan(hyb), ctx.history)
    ours = wh.lists_from(hyb, ctx, 7)
    theirs = df_.topk_from_scores(df_.ScoreMatrix(ctx.users, ctx.catalogue, raw["A"]), 7)
    assert list(ours["item_id"]) == list(theirs["item_id"])  # same ranking for every user


def test_learned_hybrid_picks_the_informative_member_and_is_perfect_out_of_sample():
    ctx = _ctx()
    data = _data(ctx)
    setting = Setting("zscore", "ridge", 0.0, "all")
    w, hyb_v, hyb_t = fit_hybrid(data, ["A", "B"], setting)
    assert w.coef[0] > 0 and w.coef[1] == 0.0 and w.n_rows == int((~ctx.history).sum())
    folds = wh.user_folds(len(ctx.users), n_folds=2)
    assert sorted(np.bincount(folds)) == [2, 2]
    assert cv_score(data, ["A", "B"], setting, folds)["ndcg@10"] == pytest.approx(1.0)


def test_group_weights_and_combine_by_group():
    ctx = _ctx()
    data = _data(ctx)
    table = data.table(["A", "B"], Setting("zscore", "ridge", 0.0, "all"))
    labels = np.array(["x", "x", "y", "y"])
    wg = wh.fit_group_weights(table, ["A", "B"], labels, ["x", "y"], "ridge", 0.0)
    assert set(wg) == {"x", "y"} and all(w.coef[0] > 0 for w in wg.values())
    hyb = wh.combine_by_group(data.scores(["A", "B"], "valid", "zscore"), wg, labels)
    assert np.array_equal(np.isnan(hyb), ctx.history)
    with pytest.raises(ValueError, match="no weights"):
        wh.combine_by_group(data.scores(["A", "B"], "valid", "zscore"), wg, np.array(["x", "x", "y", "z"]))
    with pytest.raises(ValueError, match="no regression rows"):
        wh.fit_group_weights(table, ["A", "B"], labels, ["x", "y", "z"], "ridge", 0.0)


# --------------------------------------------------------------------------------------- analyses
def test_bootstrap_interval_brackets_the_point_estimate():
    ctx = _ctx()
    data = _data(ctx)
    table = data.table(["A", "B"], Setting("zscore", "ridge", 0.0, "all"))
    w = wh.fit_weights(table, ["A", "B"], "ridge", 0.0)
    boot = bootstrap_weights(table, ["A", "B"], "ridge", 0.0, n_boot=30)
    assert boot.loc[0, "ci_low"] <= w.coef_std[0] <= boot.loc[0, "ci_high"]


def test_hit_attribution_counts_hits_and_members():
    ctx = _ctx()
    raw = _raw(ctx)
    ms = wh.MemberScores.from_raw(["A", "B"], "valid", raw, "zscore")
    w = wh.Weights(["A", "B"], np.array([1.0, 0.0]), np.array([1.0, 0.0]), "ridge", 0.0)
    hyb = wh.combine(ms, w)
    tops = [wh.member_top_lists(m, "valid", ctx, 50, raw[m]) for m in ["A", "B"]]
    per_member, counts = hit_attribution(ms, w, hyb, ctx, tops, k=2)
    assert per_member.set_index("member").loc["A", "attributed_hits"] == 6  # every relevant item (2+1+2+1) is a hit
    assert per_member.set_index("member").loc["B", "attributed_hits"] == 0
    assert counts["hits"].sum() == 6 and counts.loc[0, "hits"] == 0  # A always had them in its own top-2
    ov = member_overlap(ms, ~ctx.history, tops, k=2)
    assert len(ov) == 1 and 0.0 <= ov.loc[0, "top2_jaccard"] <= 1.0


def test_regression_table_is_centred_within_user():
    X = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 0.0], [7.0, 0.0]])
    y = np.array([1.0, 0.0, 1.0, 1.0])
    t = wh.RegressionTable.build(X, y, np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(t.X, [[-1, 0], [1, 0], [-1, 0], [1, 0]])  # user means removed per user
    np.testing.assert_allclose(t.y, [0.5, -0.5, 0.0, 0.0])
    np.testing.assert_array_equal(t.y_raw, y) and t.n_positive == 3
    sub = t.rows(np.array([True, True, False, False]))
    assert len(sub.y) == 2 and sub.n_positive == 1


def test_lists_to_scores_gives_rank_scores_and_keeps_the_mask():
    ctx = _ctx()
    raw = _raw(ctx)["A"]
    recs = df_.topk_from_scores(df_.ScoreMatrix(ctx.users, ctx.catalogue, raw), 3)
    m = wh.lists_to_scores(recs, ctx, k=3)
    assert np.array_equal(np.isnan(m), ctx.history)
    top = recs[recs.user_id == "u1"].sort_values("rank")
    cols = [list(ctx.catalogue).index(i) for i in top.item_id]
    np.testing.assert_allclose(m[0, cols], [1.0, 0.5, 1 / 3])  # reciprocal rank
    listed = np.zeros(ctx.n_items, dtype=bool)
    listed[cols] = True
    assert (m[0, ~listed & ~ctx.history[0]] == 0.0).all()  # unlisted candidates get the fill value
    lin = wh.lists_to_scores(recs, ctx, k=3, scheme="linear", fill=-1.0)
    np.testing.assert_allclose(lin[0, cols], [1.0, 2 / 3, 1 / 3])
    # a re-ranked member can be fed to the hybrid like any other member
    data = HybridData(ctx, {"A": m, "B": _raw(ctx)["B"]}, {n: wh.member_top_lists(n, "valid", ctx, 50, r)
                                                            for n, r in (("A", m), ("B", _raw(ctx)["B"]))}, ctx, {"A": m, "B": _raw(ctx)["B"]})
    w, hyb_v, _ = fit_hybrid(data, ["A", "B"], Setting("minmax", "ridge", 0.1, "union"))
    assert w.coef[0] > 0


def test_setting_parse_and_grid():
    s = Setting.parse(" rank , nnls , 0 , union ")
    assert s == Setting("rank", "nnls", 0.0, "union") and "nnls" in s.label
    assert len(default_grid()) == 30 and len(default_grid(quick=True)) == 4
