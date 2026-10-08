"""Content-based recommender using genre TF-IDF profiles (Task 1.1).

Each item is represented by its TF-IDF weighted genre vector (19 genres).
Each user profile is the rating-weighted mean of their training items' vectors.
Scores are cosine similarities between user profiles and item vectors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from project.metrics.lookups import genre_matrix, load_catalogue, load_item_genres
from project.utils.data_formats import (
    ScoreMatrix,
    load_history,
    load_split,
)

MODEL_NAME = "ContentBased"


def _tfidf_item_matrix(genres_bool: np.ndarray) -> np.ndarray:
    """Apply IDF weighting to a binary (I, G) genre matrix, return (I, G) float32."""
    n_items = genres_bool.shape[0]
    df = genres_bool.sum(axis=0).astype(np.float64)
    df = np.maximum(df, 1)
    idf = np.log(n_items / df)
    return (genres_bool.astype(np.float32) * idf.astype(np.float32))


def _user_profiles(
    train: pd.DataFrame,
    item_tfidf: np.ndarray,
    catalogue: np.ndarray,
    users: np.ndarray,
) -> np.ndarray:
    """Build (U, G) user profile matrix from training interactions.

    Each user's profile is the rating-weighted mean of their training items'
    TF-IDF vectors. Rating is normalized to [0.2, 1.0] for the 1-5 scale.
    """
    item_pos = {item: i for i, item in enumerate(catalogue)}
    n_genres = item_tfidf.shape[1]
    profiles = np.zeros((len(users), n_genres), dtype=np.float64)
    user_pos = {u: i for i, u in enumerate(users)}

    for _, row in train.iterrows():
        uid = str(row["user_id"])
        iid = str(row["item_id"])
        u_idx = user_pos.get(uid)
        i_idx = item_pos.get(iid)
        if u_idx is None or i_idx is None:
            continue
        weight = float(row["rating"]) / 5.0
        profiles[u_idx] += weight * item_tfidf[i_idx]

    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    profiles /= norms
    return profiles.astype(np.float32)


def build_content_scores(split: str) -> ScoreMatrix:
    """Compute content-based scores for all (user, item) pairs on the given split."""
    item_genres = load_item_genres()
    catalogue = load_catalogue()
    genres_bool, _ = genre_matrix(item_genres, catalogue)
    item_tfidf = _tfidf_item_matrix(genres_bool)

    item_norms = np.linalg.norm(item_tfidf, axis=1, keepdims=True)
    item_norms = np.maximum(item_norms, 1e-10)
    item_normed = item_tfidf / item_norms

    train = load_split("train")
    users = np.sort(train["user_id"].unique())

    profiles = _user_profiles(train, item_tfidf, catalogue, users)
    scores = profiles @ item_normed.T

    history = load_history(split)
    item_pos = {item: i for i, item in enumerate(catalogue)}
    for u_idx, uid in enumerate(users):
        for iid in history.get(uid, set()):
            i_idx = item_pos.get(iid)
            if i_idx is not None:
                scores[u_idx, i_idx] = np.nan

    return ScoreMatrix(user_id=users, item_id=catalogue, score=scores)
