"""The file formats every track reads and writes (agreed on day 1, owned by Track A).

Everything lives under ``results/raw/`` (git-ignored, regenerated with one command,
see ``project/models/README.md``). External (original MovieLens) ids are used
everywhere; RecBole's internal integer ids never leave ``project/models``.

* **Split** ``results/raw/splits/ml-100k.{train,valid,test}.tsv``
  columns ``user_id, item_id, rating, timestamp`` (tab-separated, ids as strings).
  Created once by ``python -m project.experiments.export_split``; nobody re-splits.

* **Score file** ``results/raw/scores/<Model>_<split>.npz`` -- one per model and
  evaluation split (``valid`` or ``test``) with the model's score for EVERY
  (user, item) pair: ``user_id`` (U,), ``item_id`` (I,), ``score`` (U, I) float32.
  Items in the user's history are ``NaN`` (not candidates): for ``valid`` the
  history is the train set, for ``test`` it is train + valid, exactly as RecBole
  masks them when it evaluates. Hybrids and re-rankers read these.

* **Recommendation list** ``results/raw/recommendations/<Name>_<split>_top50.csv``
  columns ``user_id, item_id, rank, score`` (rank 1 = best, 50 rows per user).
  Every model, hybrid and re-ranker writes this; the metrics (Track B) read it.
  The final list of 10 is simply ``rank <= 10``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, NamedTuple, Set

import numpy as np
import pandas as pd

from project.utils.paths import RESULTS_RAW_DIR

DATASET_NAME = "ml-100k"
SPLITS_DIR = RESULTS_RAW_DIR / "splits"
SCORES_DIR = RESULTS_RAW_DIR / "scores"
RECOMMENDATIONS_DIR = RESULTS_RAW_DIR / "recommendations"
CHECKPOINTS_DIR = RESULTS_RAW_DIR / "checkpoints"

SPLIT_PARTS = ("train", "valid", "test")
EVAL_SPLITS = ("valid", "test")
TOPK_FINAL = 10  # length of the list that is evaluated
TOPK_CANDIDATES = 50  # length of the exported candidate list (re-rankers pick 10 out of it)

SPLIT_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
RECOMMENDATION_COLUMNS = ["user_id", "item_id", "rank", "score"]


# ----------------------------------------------------------------------------- split
def split_path(part: str) -> Path:
    if part not in SPLIT_PARTS:
        raise ValueError(f"part must be one of {SPLIT_PARTS}, got {part!r}")
    return SPLITS_DIR / f"{DATASET_NAME}.{part}.tsv"


def save_split(df: pd.DataFrame, part: str) -> Path:
    path = split_path(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[SPLIT_COLUMNS].to_csv(path, sep="\t", index=False)
    return path


def load_split(part: str) -> pd.DataFrame:
    """Interactions of one part of the split, ids as strings."""
    return pd.read_csv(split_path(part), sep="\t", dtype={"user_id": str, "item_id": str})


def _user_item_sets(df: pd.DataFrame) -> Dict[str, Set[str]]:
    return {u: set(items) for u, items in df.groupby("user_id")["item_id"]}


def load_ground_truth(split: str) -> Dict[str, Set[str]]:
    """Relevant (held-out) items per user for ``valid`` or ``test``."""
    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {EVAL_SPLITS}, got {split!r}")
    return _user_item_sets(load_split(split))


def load_history(split: str) -> Dict[str, Set[str]]:
    """Items that must NOT be recommended when evaluating ``split``
    (train for ``valid``; train + valid for ``test``)."""
    if split not in EVAL_SPLITS:
        raise ValueError(f"split must be one of {EVAL_SPLITS}, got {split!r}")
    parts = [load_split("train")] + ([load_split("valid")] if split == "test" else [])
    return _user_item_sets(pd.concat(parts, ignore_index=True))


# ---------------------------------------------------------------------------- scores
class ScoreMatrix(NamedTuple):
    """Dense scores of one model: ``score[u, i]`` for ``user_id[u]`` and ``item_id[i]``."""

    user_id: np.ndarray  # (U,) str
    item_id: np.ndarray  # (I,) str
    score: np.ndarray  # (U, I) float32, NaN = item in the user's history

    def to_frame(self, drop_history: bool = True) -> pd.DataFrame:
        """Long format ``user_id, item_id, score`` (one row per pair)."""
        u = np.repeat(self.user_id, len(self.item_id))
        i = np.tile(self.item_id, len(self.user_id))
        df = pd.DataFrame({"user_id": u, "item_id": i, "score": self.score.ravel()})
        return df[df["score"].notna()].reset_index(drop=True) if drop_history else df


def scores_path(model: str, split: str) -> Path:
    return SCORES_DIR / f"{model}_{split}.npz"


def save_scores(sm: ScoreMatrix, model: str, split: str) -> Path:
    path = scores_path(model, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        user_id=np.asarray(sm.user_id, dtype=str),
        item_id=np.asarray(sm.item_id, dtype=str),
        score=np.asarray(sm.score, dtype=np.float32),
    )
    return path


def load_scores(model: str, split: str) -> ScoreMatrix:
    with np.load(scores_path(model, split), allow_pickle=False) as z:
        return ScoreMatrix(z["user_id"].astype(str), z["item_id"].astype(str), z["score"])


def topk_from_scores(sm: ScoreMatrix, k: int = TOPK_CANDIDATES) -> pd.DataFrame:
    """Top-k list per user from a score matrix (NaN entries are skipped).

    Convenience for hybrids / re-rankers that build their own score matrices. Note that
    the exported ``*_top50.csv`` files are produced with ``torch.topk`` (exactly like
    RecBole's evaluation), so lists re-derived here can order *tied* scores differently
    (relevant for Pop and the kNN models, whose scores have many ties).
    """
    score = np.where(np.isnan(sm.score), -np.inf, sm.score)
    # stable argsort on -score == torch.topk order for distinct values; ties -> lower index first
    order = np.argsort(-score, axis=1, kind="stable")[:, :k]
    rows = []
    for u_idx, items in enumerate(order):
        for rank, i_idx in enumerate(items, start=1):
            s = score[u_idx, i_idx]
            if not np.isfinite(s):
                break
            rows.append((sm.user_id[u_idx], sm.item_id[i_idx], rank, float(s)))
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


# ------------------------------------------------------------------- recommendations
def recommendations_path(name: str, split: str, k: int = TOPK_CANDIDATES) -> Path:
    return RECOMMENDATIONS_DIR / f"{name}_{split}_top{k}.csv"


def save_recommendations(df: pd.DataFrame, name: str, split: str, k: int = TOPK_CANDIDATES) -> Path:
    """Write a recommendation list (any model / hybrid / re-ranker) in the shared format."""
    missing = [c for c in RECOMMENDATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"recommendation list is missing columns {missing}")
    path = recommendations_path(name, split, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[RECOMMENDATION_COLUMNS].copy()
    out["user_id"] = out["user_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    out["rank"] = out["rank"].astype(int)
    out.to_csv(path, index=False, float_format="%.6g")
    return path


def load_recommendations(name: str, split: str, k: int = TOPK_CANDIDATES) -> pd.DataFrame:
    return pd.read_csv(recommendations_path(name, split, k), dtype={"user_id": str, "item_id": str})


def precision_recall_at_k(recs: pd.DataFrame, ground_truth: Dict[str, Set[str]], k: int = TOPK_FINAL):
    """Precision@k and Recall@k averaged over the users in ``recs`` that have relevant
    items. Used only as an export sanity check against RecBole's own numbers; the
    project's metrics live in ``project/metrics`` (Track B)."""
    top = recs[recs["rank"] <= k]
    p_sum = r_sum = 0.0
    n = 0
    for user, items in top.groupby("user_id")["item_id"]:
        relevant = ground_truth.get(user)
        if not relevant:
            continue
        hits = len(set(items) & relevant)
        p_sum += hits / k
        r_sum += hits / len(relevant)
        n += 1
    return p_sum / n, r_sum / n
