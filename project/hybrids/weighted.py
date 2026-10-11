"""Weighted hybrid with coefficients learned by a regression model (Task 1.3; Track C).

    score'(u, i) = sum_k w_k * z_k(u, i)

``z_k(u, i)`` is member k's score for the pair, rescaled per user (``rescale``): EASE weights,
BPR dot products, kNN similarity sums and Pop counts live on different scales and some models
have per-user offsets, so raw scores cannot be added. The weights ``w`` are learned on the
VALIDATION split only. The regression table has one row per candidate (user, movie) pair --
the union of the members' top-50 validation candidates (``candidates="union"``) or every
unseen movie (``"all"``) -- the members' rescaled scores as features and 1 if the movie is a
held-out validation interaction as target. Training interactions cannot be used (the members
have memorised them) and the test split is touched once, when the selected hybrid is applied.

Everything is aligned to an ``EvaluationContext`` (rows ``ctx.users``, columns
``ctx.catalogue``), the alignment the metrics use, so the hybrid optimises exactly what Track B
measures. Masked (history) entries are NaN throughout and stay NaN in the hybrid. The released
score files store each user's top-10 scores as NaN (an export bug, see ``_restore_top_scores``);
``aligned_scores`` restores them from the exported lists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from project.metrics import EvaluationContext
from project.metrics.evaluate import index_matrix
from project.utils.data_formats import (TOPK_CANDIDATES, TOPK_FINAL, ScoreMatrix, load_recommendations,
                                        load_scores, recommendations_path, scores_path, topk_from_scores)

DEFAULT_MEMBERS = ["ItemKNN", "EASE", "BPR", "NeuMF", "LightGCN"]  # one model per lecture family (the plan's shortlist)
BASELINES = ("Random", "Pop")
RESCALERS = ("zscore", "minmax", "rank", "none")
REGRESSIONS = ("ridge", "nnls", "logistic")
CANDIDATE_RULES = ("union", "all")
N_FOLDS, SEED = 5, 2020


# ------------------------------------------------------------------------------------------ inputs
def aligned_scores(name: str, split: str, ctx: EvaluationContext) -> np.ndarray:
    """(U, I) float32 scores of ``name`` aligned to ``ctx.users`` x ``ctx.catalogue``; NaN = history.

    Raises when the file does not cover the context, or when its NaN pattern is not the masked
    history of ``split`` (a list of the other split, or another protocol, was exported).
    """
    sm = load_scores(name, split)
    u = pd.Index(sm.user_id).get_indexer(ctx.users)
    i = pd.Index(sm.item_id).get_indexer(ctx.catalogue)
    if (u < 0).any() or (i < 0).any():
        raise ValueError(f"{scores_path(name, split)} lacks {int((u < 0).sum())} users / {int((i < 0).sum())} "
                         f"items of the {split} context")
    m = sm.score[u][:, i].astype(np.float32)
    nan = np.isnan(m)
    if (ctx.history & ~nan).any():
        raise ValueError(f"{scores_path(name, split)}: history items are not NaN -- is this a {split!r} matrix?")
    extra = nan & ~ctx.history
    return _restore_top_scores(m, extra, name, split, ctx) if extra.any() else m


def _restore_top_scores(m: np.ndarray, extra: np.ndarray, name: str, split: str, ctx: EvaluationContext) -> np.ndarray:
    """Put back the top-10 scores that Track A's release stores as NaN.

    ``pipeline.scores_and_topk`` took the stored matrix as ``scores.float().cpu().numpy()``, which
    for a float32 CPU tensor is a *view*, and then blanked the top-10 of that tensor with
    ``scatter_`` to find ranks 11-50 -- so every user's 10 best scores are NaN in the file (Pop,
    whose scores are float64, got a copy and is intact). Ranks 1-10 of the exported list carry
    exactly those scores; any other NaN outside the masked history is an error.
    """
    if not recommendations_path(name, split).exists():
        raise ValueError(f"{scores_path(name, split)}: {int(extra.sum())} NaN entries outside the masked history "
                         "and no exported list to restore them from")
    top = load_recommendations(name, split)
    top = top[top["rank"] <= TOPK_FINAL]
    rows = pd.Index(ctx.users).get_indexer(top["user_id"].astype(str))
    cols = pd.Index(ctx.catalogue).get_indexer(top["item_id"].astype(str))
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError(f"{recommendations_path(name, split)} has users or items outside the {split} context")
    known = np.zeros(extra.shape, dtype=bool)
    known[rows, cols] = True
    if (extra & ~known).any():
        raise ValueError(f"{scores_path(name, split)}: {int((extra & ~known).sum())} NaN entries outside the masked "
                         f"history that are not the exported top-{TOPK_FINAL}")
    out = m.copy()
    out[rows, cols] = top["score"].to_numpy(dtype=np.float32)
    return out


def rescale(m: np.ndarray, method: str) -> np.ndarray:
    """Rescale every user's scores over that user's candidate (non-NaN) movies; NaN stays NaN.

    zscore: (s - mean_u) / std_u (0 when a user's scores are constant); minmax: (s - min_u) /
    (max_u - min_u); rank: percentile rank within the user's candidates in (0, 1], ties averaged
    (robust to the many tied scores of Pop and the kNN models); none: unchanged.
    """
    if method not in RESCALERS:
        raise ValueError(f"rescale must be one of {RESCALERS}, got {method!r}")
    m = np.asarray(m, dtype=np.float32)
    if method == "none":
        return m.copy()
    valid = ~np.isnan(m)
    out = np.full(m.shape, np.nan, dtype=np.float32)
    if method == "rank":
        for r in range(m.shape[0]):
            v = valid[r]
            if v.any():
                out[r, v] = stats.rankdata(m[r, v], method="average") / v.sum()
        return out
    with np.errstate(invalid="ignore"):
        if method == "zscore":
            mu, sd = np.nanmean(m, axis=1, keepdims=True), np.nanstd(m, axis=1, keepdims=True)
            out = ((m - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)
        else:  # minmax
            lo, hi = np.nanmin(m, axis=1, keepdims=True), np.nanmax(m, axis=1, keepdims=True)
            out = ((m - lo) / np.where(hi > lo, hi - lo, 1.0)).astype(np.float32)
    return out


@dataclass
class MemberScores:
    """Rescaled scores of the members on one split: ``z[k, u, i]`` (NaN = masked history)."""

    members: List[str]
    split: str
    rescale: str
    z: np.ndarray  # (K, U, I) float32

    @classmethod
    def from_raw(cls, members: Sequence[str], split: str, raw: Dict[str, np.ndarray], method: str) -> "MemberScores":
        return cls(list(members), split, method, np.stack([rescale(raw[m], method) for m in members]))

    def subset(self, users: np.ndarray) -> "MemberScores":
        return MemberScores(self.members, self.split, self.rescale, self.z[:, users, :])


def member_top_lists(name: str, split: str, ctx: EvaluationContext, k: int = TOPK_CANDIDATES,
                     raw: Optional[np.ndarray] = None) -> np.ndarray:
    """(U, k) catalogue indices of the member's exported top-k list aligned to ``ctx.users`` (-1 =
    empty slot). Falls back to the score matrix when no list file was exported for ``name``."""
    if recommendations_path(name, split).exists():
        recs = load_recommendations(name, split)
    else:
        if raw is None:
            raw = aligned_scores(name, split, ctx)
        recs = topk_from_scores(ScoreMatrix(ctx.users, ctx.catalogue, raw), k)
    idx, _ = index_matrix(recs, ctx, k)
    return idx


def lists_to_scores(recs: pd.DataFrame, ctx: EvaluationContext, k: int = TOPK_CANDIDATES, scheme: str = "reciprocal",
                    fill: float = 0.0) -> np.ndarray:
    """(U, I) score matrix built from a recommendation list -- the input the hybrid needs when a member
    has been re-ranked first (Task 3.3, "re-rank then combine"): a re-ranked list has positions, not
    comparable scores, so rank r gets 1/r (``reciprocal``) or (k + 1 - r) / k (``linear``); movies
    outside the list get ``fill`` (0 = below every listed movie) and the masked history stays NaN.
    The result can be put in ``HybridData``'s ``raw`` dictionaries like any member's matrix."""
    if scheme not in ("reciprocal", "linear"):
        raise ValueError(f"scheme must be 'reciprocal' or 'linear', got {scheme!r}")
    idx, _ = index_matrix(recs, ctx, k)
    out = np.where(ctx.history, np.nan, float(fill)).astype(np.float32)
    ranks = np.arange(1, k + 1, dtype=np.float32)
    values = 1.0 / ranks if scheme == "reciprocal" else (k + 1 - ranks) / k
    filled = idx >= 0
    rows = np.broadcast_to(np.arange(len(idx))[:, None], idx.shape)
    out[rows[filled], idx[filled]] = np.broadcast_to(values, idx.shape)[filled]
    return out


def candidate_mask(ctx: EvaluationContext, rule: str, member_lists: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """(U, I) bool: the (user, movie) pairs that form the regression table.

    ``all``: every movie outside the user's masked history; ``union``: every movie in at least
    one member's top-50 candidate list (the region that decides the final top-10, with the
    members' hard negatives).
    """
    if rule not in CANDIDATE_RULES:
        raise ValueError(f"candidates must be one of {CANDIDATE_RULES}, got {rule!r}")
    if rule == "all":
        return ~ctx.history
    if not member_lists:
        raise ValueError("candidates='union' needs the members' top-k lists")
    mask = np.zeros(ctx.history.shape, dtype=bool)
    for idx in member_lists:
        filled = idx >= 0
        rows = np.broadcast_to(np.arange(len(idx))[:, None], idx.shape)
        mask[rows[filled], idx[filled]] = True
    if (mask & ctx.history).any():
        raise ValueError("a member list contains masked history items")
    return mask


# ------------------------------------------------------------------------------- regression table
def _centre_by_user(a: np.ndarray, user: np.ndarray) -> np.ndarray:
    """Subtract every user's mean (over that user's rows) from a (n,) or (n, K) array."""
    n_users = int(user.max()) + 1
    counts = np.bincount(user, minlength=n_users).astype(float)
    counts = np.where(counts > 0, counts, 1.0)
    if a.ndim == 1:
        return a - (np.bincount(user, weights=a, minlength=n_users) / counts)[user]
    means = np.stack([np.bincount(user, weights=a[:, k], minlength=n_users) for k in range(a.shape[1])], axis=1)
    return a - (means / counts[:, None])[user]


@dataclass
class RegressionTable:
    """One row per candidate (user, movie) pair.

    ``X`` (n, K) and ``y`` (n,) are centred WITHIN USER (every user's rows have mean zero), so a
    regression on them is a fixed-effects regression: the weights explain which of a user's
    candidates are relevant, not how many relevant items a user has -- the only thing a per-user
    ranking can use. Without this, members whose candidates are popular for everybody (Pop, FISM)
    pick up the between-user base rate. ``y_raw`` keeps the 0/1 targets (logistic regression),
    ``user`` (n,) the row index of the pair's user in ``ctx.users`` (rows are user-sorted).
    """

    X: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    user: np.ndarray

    @classmethod
    def build(cls, X: np.ndarray, y: np.ndarray, user: np.ndarray) -> "RegressionTable":
        X, y, user = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(user)
        return cls(_centre_by_user(X, user), _centre_by_user(y, user), y, user)

    @property
    def n_positive(self) -> int:
        return int(self.y_raw.sum())

    def rows(self, keep: np.ndarray) -> "RegressionTable":
        """Subset of rows; valid when ``keep`` selects whole users (folds, bootstrap, groups)."""
        return RegressionTable(self.X[keep], self.y[keep], self.y_raw[keep], self.user[keep])


def regression_table(ms: MemberScores, ctx: EvaluationContext, cand: np.ndarray) -> RegressionTable:
    uu, ii = np.nonzero(cand)
    X = ms.z[:, uu, ii].T.astype(np.float64)
    if np.isnan(X).any():
        raise ValueError("candidate pairs include masked (NaN) scores")
    return RegressionTable.build(X, ctx.relevant[uu, ii].astype(np.float64), uu)


# ------------------------------------------------------------------------------------------ fitting
@dataclass
class Weights:
    """The learned coefficients of one weighted hybrid.

    ``coef`` is on the rescaled-feature scale and is what ``combine`` uses; ``coef_std`` is the
    coefficient per standard deviation of the feature (comparable between members and
    rescalings); ``share`` normalises ``coef_std`` to sum to one (the lecture's sum w_k = 1),
    NaN when the sum is not positive.
    """

    members: List[str]
    coef: np.ndarray
    coef_std: np.ndarray
    regression: str
    lam: float
    rescale: str = ""
    candidates: str = ""
    n_rows: int = 0
    n_positive: int = 0

    @property
    def share(self) -> np.ndarray:
        total = self.coef_std.sum()
        return self.coef_std / total if total > 0 else np.full(len(self.coef_std), np.nan)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"member": self.members, "coef": self.coef, "coef_std": self.coef_std, "share": self.share})

    def subset(self, members: Sequence[str]) -> "Weights":
        pos = [self.members.index(m) for m in members]
        return Weights(list(members), self.coef[pos], self.coef_std[pos], self.regression, self.lam, self.rescale,
                       self.candidates, self.n_rows, self.n_positive)


def _logistic(Xs: np.ndarray, y: np.ndarray, lam: float, maxiter: int = 500) -> np.ndarray:
    """L2-regularised logistic regression (intercept unpenalised), L-BFGS with analytic gradient."""
    n, k = Xs.shape

    def f(p):
        w, b = p[:k], p[k]
        z = Xs @ w + b
        loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * w @ w
        r = (special.expit(z) - y) / n
        return loss, np.concatenate([Xs.T @ r + lam * w, [r.sum()]])

    res = optimize.minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x[:k]


def fit_weights(table: RegressionTable, members: Sequence[str], regression: str, lam: float = 0.0,
                rescale: str = "", candidates: str = "") -> Weights:
    """Fit the regression on the table (features standardised internally, so ``lam`` means the same
    for every rescaling) and return the weights on the original feature scale.

    ridge: closed form (X'X/n + lam I) w = X'y/n on centred data; nnls: non-negative least
    squares (ridge-augmented when lam > 0); logistic: L2-regularised logistic regression.
    """
    if regression not in REGRESSIONS:
        raise ValueError(f"regression must be one of {REGRESSIONS}, got {regression!r}")
    X = table.X  # within-user centred features
    y = table.y_raw if regression == "logistic" else table.y  # 0/1 for logistic, within-user centred otherwise
    n, k = X.shape
    if n == 0 or k == 0:
        raise ValueError("empty regression table")
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    yc = y - y.mean()
    if regression == "ridge":
        if lam > 0:
            ws = np.linalg.solve(Xs.T @ Xs / n + lam * np.eye(k), Xs.T @ yc / n)
        else:  # plain least squares; minimum-norm solution when a feature is constant or collinear
            ws = np.linalg.lstsq(Xs, yc, rcond=None)[0]
    elif regression == "nnls":
        Xa = np.vstack([Xs, np.sqrt(n * lam) * np.eye(k)]) if lam > 0 else Xs
        ya = np.concatenate([yc, np.zeros(k)]) if lam > 0 else yc
        ws, _ = optimize.nnls(Xa, ya)
    else:
        ws = _logistic(Xs, y, lam)
    return Weights(list(members), ws / sd, ws, regression, lam, rescale, candidates, n, table.n_positive)


def fit_group_weights(table: RegressionTable, members: Sequence[str], user_labels: np.ndarray,
                      labels: Sequence[str], regression: str, lam: float = 0.0, **tags) -> Dict[str, Weights]:
    """One ``Weights`` per user group (Task 2.6): the regression is fitted on the rows of that
    group's users only. ``user_labels`` (U,) holds the group label of every user of the context."""
    row_label = np.asarray(user_labels)[table.user]
    out = {}
    for g in labels:
        keep = row_label == g
        if not keep.any():
            raise ValueError(f"no regression rows for group {g!r}")
        out[g] = fit_weights(table.rows(keep), members, regression, lam, **tags)
    return out


# ---------------------------------------------------------------------------------------- combining
def combine(ms: MemberScores, w: Weights) -> np.ndarray:
    """(U, I) hybrid scores sum_k w_k z_k; NaN where the history is masked."""
    if list(w.members) != list(ms.members):
        raise ValueError(f"weights are for {w.members}, scores for {ms.members}")
    return np.tensordot(w.coef.astype(np.float32), ms.z, axes=1)


def combine_by_group(ms: MemberScores, weights_by_group: Dict[str, Weights], user_labels: np.ndarray) -> np.ndarray:
    """Like ``combine`` but every user gets the weights of their group."""
    out = np.full(ms.z.shape[1:], np.nan, dtype=np.float32)
    user_labels = np.asarray(user_labels)
    missing = set(np.unique(user_labels)) - set(weights_by_group)
    if missing:
        raise ValueError(f"no weights for groups {sorted(missing)}")
    for g, w in weights_by_group.items():
        rows = user_labels == g
        if rows.any():
            out[rows] = combine(ms.subset(rows), w)
    return out


def score_matrix(hyb: np.ndarray, ctx: EvaluationContext) -> ScoreMatrix:
    return ScoreMatrix(ctx.users, ctx.catalogue, hyb.astype(np.float32))


def lists_from(hyb: np.ndarray, ctx: EvaluationContext, k: int = TOPK_CANDIDATES) -> pd.DataFrame:
    """Top-k list per user in the shared format (``user_id, item_id, rank, score``)."""
    return topk_from_scores(score_matrix(hyb, ctx), k)


# -------------------------------------------------------------------------------- cross-fitting
def user_folds(n_users: int, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """(U,) fold id per user: the weights are learned on validation, so every selection among
    settings is made on users whose validation interactions the weights never saw."""
    return np.random.default_rng(seed).permutation(n_users) % n_folds


def cross_fitted(ms: MemberScores, table: RegressionTable, folds: np.ndarray, regression: str, lam: float = 0.0,
                 user_labels: Optional[np.ndarray] = None, labels: Optional[Sequence[str]] = None) -> np.ndarray:
    """(U, I) hybrid scores where each user's row comes from weights fitted WITHOUT that user's
    fold (global weights, or per-group weights when ``user_labels`` is given)."""
    out = np.full(ms.z.shape[1:], np.nan, dtype=np.float32)
    for f in np.unique(folds):
        sub = table.rows(folds[table.user] != f)
        held = folds == f
        if user_labels is None:
            out[held] = combine(ms.subset(held), fit_weights(sub, ms.members, regression, lam))
        else:
            wg = fit_group_weights(sub, ms.members, user_labels, labels, regression, lam)
            out[held] = combine_by_group(ms.subset(held), wg, np.asarray(user_labels)[held])
    return out
