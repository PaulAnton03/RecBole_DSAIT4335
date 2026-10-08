"""Hand-computed checks of the re-rankers (run: python -m pytest project/tests -q)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from project.metrics import EvaluationContext, calibration, evaluate, lookups, popularity
from project.rerankers import (MMR, OBJECTIVES, Calibrated, PopCalibrated, XQuAD, greedy_order, knee,
                               list_reranker, make_objective, normalise_scores, rerank, reranked_name, sweep)

# ------------------------------------------- worked example: 5 candidates, pick in order, lam = 0.5
EXAMPLE_GENRES = {"A": ("Action", "Sci-Fi"), "B": ("Action", "Sci-Fi"), "C": ("Action", "Adventure", "Sci-Fi"),
                  "D": ("Crime", "Drama"), "E": ("Animation", "Comedy"), "H": ("Drama",)}
EXAMPLE_REL = np.array([1.0, 0.9, 0.85, 0.7, 0.35])  # A..E, already in [0, 1]


def example_ctx():
    return EvaluationContext.build({"u": {"A"}}, {"u": {"H"}}, EXAMPLE_GENRES)


def state_for(objective, ctx, items, user_row=0):
    objective.prepare(ctx)
    return objective.user_state(user_row, np.array([ctx.item_pos[i] for i in items]))


def test_mmr_gain_is_mean_distance_to_placed_items():
    state = state_for(MMR(), example_ctx(), "ABCDE")
    np.testing.assert_allclose(state.gains(), 0.0)  # first slot: nothing to be diverse from
    state.add(0)  # A
    np.testing.assert_allclose(state.gains()[1:], [0.0, 1 / 3, 1.0, 1.0])
    state.add(3)  # D
    np.testing.assert_allclose(state.gains()[[1, 2, 4]], [0.5, 2 / 3, 1.0])


def test_mmr_reproduces_hand_example():
    order = greedy_order(EXAMPLE_REL, state_for(MMR(), example_ctx(), "ABCDE"), lam=0.5)
    # slot 2: D 0.85 > E 0.675 > C 0.592 > B 0.45;  slot 3: C 0.758 > B 0.70 > E 0.675;
    # slot 4: E 0.675 > B 0.45 + 0.5 * 4/9 = 0.672
    assert "".join("ABCDE"[j] for j in order) == "ADCEB"
    assert "".join("ABCDE"[j] for j in greedy_order(EXAMPLE_REL, state_for(MMR(), example_ctx(), "ABCDE"), 0.0)) == "ABCDE"


def test_greedy_depth_only_reranks_the_top_slots():
    full = greedy_order(EXAMPLE_REL, state_for(MMR(), example_ctx(), "ABCDE"), lam=0.5)
    top2 = greedy_order(EXAMPLE_REL, state_for(MMR(), example_ctx(), "ABCDE"), lam=0.5, depth=2)
    np.testing.assert_array_equal(top2[:2], full[:2])
    np.testing.assert_array_equal(top2[2:], [1, 2, 4])  # B, C, E keep their original order


def test_normalise_scores():
    np.testing.assert_allclose(normalise_scores([3.0, 1.0, 2.0]), [1.0, 0.0, 0.5])
    np.testing.assert_allclose(normalise_scores([2.0, 2.0]), [1.0, 1.0])
    with pytest.raises(ValueError):
        normalise_scores([1.0, np.nan])


# ------------------------------------------------------------- objectives = the reported metrics
def test_calibration_gain_is_minus_the_reported_miscalibration():
    ctx = EvaluationContext.build({"u": {"A"}}, {"u": {"H", "D"}}, EXAMPLE_GENRES)  # history: Drama, Crime/Drama
    cand = np.array([ctx.item_pos[i] for i in "ABCE"])
    state = state_for(Calibrated(), ctx, "ABCE")
    state.add(0)  # A placed
    for j in (1, 2, 3):
        mc = calibration.miscalibration(cand[[0, j]][None, :], ctx.train, ctx.p_gi)[0]
        assert state.gains()[j] == pytest.approx(-mc)


def test_popularity_gain_is_minus_the_reported_upd():
    ctx = EvaluationContext.build({"u": {"A"}}, {"u": {"H", "D"}}, EXAMPLE_GENRES)
    groups = np.array([lookups.HEAD, lookups.HEAD, lookups.MID, lookups.TAIL, lookups.UNSEEN, lookups.TAIL])  # A B C D E H
    ctx = replace(ctx, item_groups=groups)
    cand = np.array([ctx.item_pos[i] for i in "ABCE"])
    state = state_for(PopCalibrated(), ctx, "ABCE")
    state.add(1)  # B placed
    for j in (0, 2, 3):
        upd = popularity.user_popularity_deviation(cand[[1, j]][None, :], ctx.train, ctx.item_groups)[0]
        assert state.gains()[j] == pytest.approx(-upd)


def test_xquad_bonus_shrinks_as_a_category_fills():
    ctx = replace(example_ctx(), item_groups=np.array([lookups.HEAD, lookups.HEAD, lookups.MID, lookups.TAIL,
                                                       lookups.UNSEEN, lookups.HEAD]))  # A B C D E H
    state = state_for(XQuAD(), ctx, "ABCDE")  # long tail = D (tail) and E (unseen)
    np.testing.assert_allclose(state.gains(), 0.5)  # empty list: both categories equally welcome
    state.add(0)  # A (head): the short head is full, the long tail is missing
    np.testing.assert_allclose(state.gains(), [0.0, 0.0, 0.0, 0.5, 0.5])
    state.add(3)  # D (tail): half / half
    np.testing.assert_allclose(state.gains(), 0.25)
    personal = state_for(XQuAD(personalised=True), ctx, "ABCDE")  # the user's history {H} is all short head
    np.testing.assert_allclose(personal.gains(), [1.0, 1.0, 1.0, 0.0, 0.0])


# ------------------------------------------------------------------- whole lists (random data)
def random_setup(seed=0, n_items=40, n_users=8, n_cand=15):
    """Popularity-skewed toy data whose candidate lists favour popular items, like the real models."""
    rng = np.random.default_rng(seed)
    names = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]
    items = [f"i{j}" for j in range(n_items)]  # i0 = most popular
    item_genres = {it: tuple(str(g) for g in rng.choice(names, size=rng.integers(1, 3), replace=False)) for it in items}
    p = 1.0 / np.arange(1, n_items + 1)
    train, truth, rows = {}, {}, []
    for u in range(n_users):
        user = f"u{u}"
        train[user] = {str(i) for i in rng.choice(items, size=8, replace=False, p=p / p.sum())}
        rest = [it for it in items if it not in train[user]]
        truth[user] = {str(i) for i in rng.choice(rest, size=3, replace=False)}
        cand = sorted(rng.choice(rest, size=n_cand, replace=False), key=lambda it: int(it[1:]))  # popular first
        scores = np.sort(rng.random(n_cand))[::-1]
        rows += [(user, str(it), r + 1, s) for r, (it, s) in enumerate(zip(cand, scores))]
    ctx = EvaluationContext.build(truth, train, item_genres)
    return ctx, pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"])


@pytest.mark.parametrize("name", list(OBJECTIVES))
def test_lambda_zero_keeps_the_model_order(name):
    ctx, recs = random_setup()
    out = rerank(recs, make_objective(name), 0.0, ctx)
    pd.testing.assert_frame_equal(out[["user_id", "item_id", "rank"]], recs[["user_id", "item_id", "rank"]])


@pytest.mark.parametrize("name", list(OBJECTIVES))
def test_output_is_a_reordering_in_the_shared_format(name):
    ctx, recs = random_setup()
    out = rerank(recs, make_objective(name), 0.7, ctx)
    assert list(out.columns) == ["user_id", "item_id", "rank", "score"]
    for user, cand in recs.groupby("user_id"):
        got = out[out.user_id == user]
        assert set(got.item_id) == set(cand.item_id)
        np.testing.assert_array_equal(got["rank"], np.arange(1, len(cand) + 1))
        assert got["score"].is_monotonic_decreasing and got["score"].iloc[0] == 1.0


@pytest.mark.parametrize("name", list(OBJECTIVES))
def test_each_objective_improves_its_own_metric(name):
    ctx, recs = random_setup()
    objective = make_objective(name)
    key = f"{objective.target_metric}@5"
    before = evaluate(recs, ctx, k=5).summary[key]
    after = evaluate(rerank(recs, objective, 1.0, ctx), ctx, k=5).summary[key]
    higher_is_better = objective.target_metric in ("ild", "tailshare")
    assert (after > before) if higher_is_better else (after < before)


def test_list_reranker_keeps_k_out_items_from_the_top_k_in():
    ctx, recs = random_setup()
    cols = ["user_id", "item_id", "rank"]
    out = list_reranker(MMR(), 0.6, ctx)(recs, k_out=5, k_in=10)
    top_in = recs[recs["rank"] <= 10]
    assert out.groupby("user_id").size().eq(5).all()
    expected = rerank(top_in, MMR(), 0.6, ctx)
    pd.testing.assert_frame_equal(out[cols], expected[expected["rank"] <= 5][cols].reset_index(drop=True))
    same = list_reranker(MMR(), 0.0, ctx)(recs, k_out=5, k_in=10)  # lam = 0: the list's own top 5
    pd.testing.assert_frame_equal(same[cols], recs[recs["rank"] <= 5][cols].reset_index(drop=True))


def test_rerank_rejects_bad_input():
    ctx, recs = random_setup()
    with pytest.raises(ValueError):
        rerank(recs, MMR(), 1.5, ctx)
    with pytest.raises(ValueError):
        rerank(recs.assign(user_id="nobody"), MMR(), 0.5, ctx)


# -------------------------------------------------------------------------------- trade-off
def test_knee_on_a_concave_curve():
    curve = pd.DataFrame({"lambda": [0.0, 0.25, 0.5, 0.75, 1.0],
                          "ndcg@10": [0.25, 0.245, 0.23, 0.18, 0.10],
                          "ild@10": [0.70, 0.78, 0.80, 0.81, 0.82],
                          "miscalibration@10": [0.70, 0.62, 0.60, 0.59, 0.58]})
    # x = [0, .67, .83, .92, 1], y = [1, .97, .87, .53, 0]: furthest above the line x + y = 1 at lam = 0.5
    assert knee(curve, "ild") == 0.5
    assert knee(curve, "miscalibration") == 0.5  # lower is better: same shape
    assert knee(curve.assign(**{"ild@10": 0.70}), "ild") == 0.0  # no improvement: do not re-rank


def test_sweep_starts_at_the_model_itself():
    ctx, recs = random_setup()
    curve = sweep(recs, MMR(), ctx, lambdas=(0.0, 0.5, 1.0), k=5, name="Toy")
    assert list(curve["lambda"]) == [0.0, 0.5, 1.0] and set(curve["objective"]) == {"MMR"}
    assert curve["ndcg@5"].iloc[0] == pytest.approx(evaluate(recs, ctx, k=5).summary["ndcg@5"])
    assert curve["ild@5"].iloc[-1] > curve["ild@5"].iloc[0]


def test_names():
    assert reranked_name("EASE", MMR()) == "EASE+MMR"
    with pytest.raises(ValueError):
        make_objective("Unknown")
