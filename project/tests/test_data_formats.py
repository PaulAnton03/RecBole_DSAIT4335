"""Unit tests for the shared file formats (run: python -m pytest project/tests -q)."""
import numpy as np
import pandas as pd
import pytest

from project.utils import data_formats as df_


def test_topk_from_scores_skips_history_and_orders_by_score():
    sm = df_.ScoreMatrix(
        user_id=np.array(["u1", "u2"]),
        item_id=np.array(["a", "b", "c", "d"]),
        score=np.array([[0.1, np.nan, 0.9, 0.5], [np.nan, np.nan, np.nan, 1.0]], dtype=np.float32),
    )
    recs = df_.topk_from_scores(sm, k=3)
    u1 = recs[recs.user_id == "u1"]
    assert list(u1.item_id) == ["c", "d", "a"]
    assert list(u1["rank"]) == [1, 2, 3]
    u2 = recs[recs.user_id == "u2"]
    assert list(u2.item_id) == ["d"]  # only one candidate left after masking history


def test_precision_recall_at_k_matches_hand_computation():
    recs = pd.DataFrame({
        "user_id": ["u1"] * 3 + ["u2"] * 3,
        "item_id": ["a", "b", "c", "x", "y", "z"],
        "rank": [1, 2, 3, 1, 2, 3],
        "score": [3, 2, 1, 3, 2, 1],
    })
    gt = {"u1": {"a", "c", "q"}, "u2": {"y"}, "u3": {"m"}}  # u3 has no list -> ignored
    p, r = df_.precision_recall_at_k(recs, gt, k=3)
    assert p == pytest.approx(((2 / 3) + (1 / 3)) / 2)
    assert r == pytest.approx(((2 / 3) + 1.0) / 2)


def test_score_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(df_, "SCORES_DIR", tmp_path)
    sm = df_.ScoreMatrix(np.array(["1", "2"]), np.array(["10", "20", "30"]),
                         np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, 1.0]], dtype=np.float32))
    df_.save_scores(sm, "Dummy", "test")
    back = df_.load_scores("Dummy", "test")
    assert list(back.user_id) == ["1", "2"] and list(back.item_id) == ["10", "20", "30"]
    np.testing.assert_array_equal(np.isnan(back.score), np.isnan(sm.score))
    long = back.to_frame()
    assert len(long) == 4 and set(long.columns) == {"user_id", "item_id", "score"}


def test_recommendations_round_trip_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(df_, "RECOMMENDATIONS_DIR", tmp_path)
    recs = pd.DataFrame({"user_id": [1, 1], "item_id": [5, 6], "rank": [1, 2], "score": [0.5, 0.25]})
    df_.save_recommendations(recs, "Dummy", "valid", k=2)
    back = df_.load_recommendations("Dummy", "valid", k=2)
    assert back.dtypes["user_id"] == object and list(back.item_id) == ["5", "6"]
    with pytest.raises(ValueError):
        df_.save_recommendations(recs.drop(columns=["rank"]), "Dummy", "valid", k=2)
