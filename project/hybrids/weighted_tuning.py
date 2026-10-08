"""Settings of the weighted hybrid and their cross-validated selection (Task 1.5; Track C).

A setting = per-user rescaling x regression model x regularisation x candidate rule. Because
the weights themselves are learned on the validation split, a setting is scored by 5-fold
cross-fitting over users on that split: every user's hybrid list comes from weights that never
saw the user's validation interactions. The score is Track B's NDCG@10 (``project.metrics``).
The test split is not touched here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from project.metrics import EvaluationContext, evaluate
from project.hybrids.weighted import (MemberScores, candidate_mask, cross_fitted, lists_from, regression_table,
                                      user_folds, TOPK_FINAL)

LAMBDAS = {"ridge": [1e-3, 1e-1], "nnls": [0.0], "logistic": [1e-4, 1e-2]}


@dataclass(frozen=True)
class Setting:
    rescale: str
    regression: str
    lam: float
    candidates: str

    @property
    def label(self) -> str:
        return f"{self.rescale}/{self.regression}(lam={self.lam:g})/{self.candidates}"

    @classmethod
    def parse(cls, text: str) -> "Setting":
        r, g, lam, c = [t.strip() for t in text.split(",")]
        return cls(r, g, float(lam), c)


def default_grid(quick: bool = False) -> List[Setting]:
    rescales = ["zscore", "rank"] if quick else ["zscore", "minmax", "rank"]
    regressions = ["ridge"] if quick else ["ridge", "nnls", "logistic"]
    candidates = ["union"] if quick else ["union", "all"]
    return [Setting(r, g, lam, c) for c in candidates for r in rescales for g in regressions for lam in LAMBDAS[g]]


class HybridData:
    """The validation (and test) inputs of every member, with caches of the rescaled scores and
    candidate masks so that a grid of settings reuses them."""

    def __init__(self, ctx_valid: EvaluationContext, raw_valid: Dict[str, np.ndarray], lists_valid: Dict[str, np.ndarray],
                 ctx_test: Optional[EvaluationContext] = None, raw_test: Optional[Dict[str, np.ndarray]] = None):
        self.ctx = {"valid": ctx_valid, "test": ctx_test}
        self.raw = {"valid": raw_valid, "test": raw_test or {}}
        self.lists_valid = lists_valid
        self._z: Dict[tuple, MemberScores] = {}
        self._cand: Dict[tuple, np.ndarray] = {}

    def scores(self, members: Sequence[str], split: str, method: str) -> MemberScores:
        key = (tuple(members), split, method)
        if key not in self._z:
            self._z[key] = MemberScores.from_raw(members, split, self.raw[split], method)
        return self._z[key]

    def candidates(self, members: Sequence[str], rule: str) -> np.ndarray:
        key = (tuple(members) if rule == "union" else (), rule)
        if key not in self._cand:
            self._cand[key] = candidate_mask(self.ctx["valid"], rule, [self.lists_valid[m] for m in members])
        return self._cand[key]

    def table(self, members: Sequence[str], setting: Setting):
        return regression_table(self.scores(members, "valid", setting.rescale), self.ctx["valid"],
                                self.candidates(members, setting.candidates))


def cv_score(data: HybridData, members: Sequence[str], setting: Setting, folds: np.ndarray, k: int = TOPK_FINAL,
             user_labels: Optional[np.ndarray] = None, labels: Optional[Sequence[str]] = None) -> pd.Series:
    """Metrics of the cross-fitted validation lists of one setting (``evaluate`` summary)."""
    ms = data.scores(members, "valid", setting.rescale)
    hyb = cross_fitted(ms, data.table(members, setting), folds, setting.regression, setting.lam, user_labels, labels)
    return evaluate(lists_from(hyb, data.ctx["valid"], k), data.ctx["valid"], k, setting.label).summary


def tune(data: HybridData, members: Sequence[str], grid: Sequence[Setting], folds: Optional[np.ndarray] = None,
         k: int = TOPK_FINAL, verbose: bool = True):
    """Score every setting of the grid; returns (table sorted by CV NDCG@10, best setting).
    Ties keep the grid order, which lists the simpler settings first."""
    folds = user_folds(len(data.ctx["valid"].users)) if folds is None else folds
    rows = []
    for i, s in enumerate(grid):
        t0 = time.time()
        summary = cv_score(data, members, s, folds, k)
        table = data.table(members, s)
        rows.append({"rescale": s.rescale, "regression": s.regression, "lam": s.lam, "candidates": s.candidates,
                     f"cv_ndcg@{k}": summary[f"ndcg@{k}"], f"cv_recall@{k}": summary[f"recall@{k}"],
                     f"cv_precision@{k}": summary[f"precision@{k}"], "rows": len(table.y), "positives": table.n_positive,
                     "seconds": round(time.time() - t0, 1), "order": i})
        if verbose:
            print(f"  [{i + 1:2d}/{len(grid)}] {s.label:34s} CV NDCG@{k} {summary[f'ndcg@{k}']:.4f}  ({rows[-1]['seconds']}s)", flush=True)
    out = pd.DataFrame(rows).sort_values([f"cv_ndcg@{k}", "order"], ascending=[False, True]).reset_index(drop=True)
    best = out.iloc[0]
    return out.drop(columns="order"), Setting(best["rescale"], best["regression"], float(best["lam"]), best["candidates"])


def fit_hybrid(data: HybridData, members: Sequence[str], setting: Setting):
    """Fit the weights on the whole validation split and apply them: returns
    ``(weights, hybrid scores on valid, hybrid scores on test)`` (test is None without test data).
    This is the function Tracks D and E reuse (combine first, re-rank afterwards; or re-rank the
    members first and feed the re-ranked scores in as ``raw``)."""
    from project.hybrids.weighted import combine, fit_weights

    w = fit_weights(data.table(members, setting), members, setting.regression, setting.lam,
                    rescale=setting.rescale, candidates=setting.candidates)
    hyb_valid = combine(data.scores(members, "valid", setting.rescale), w)
    hyb_test = combine(data.scores(members, "test", setting.rescale), w) if data.raw["test"] else None
    return w, hyb_valid, hyb_test
