"""The accuracy / beyond-accuracy trade-off of a re-ranker (Task 3.2, Track E).

``sweep`` re-ranks one model's lists for a grid of lam values and evaluates each with Track B's
``evaluate`` (validation split: lam is a hyper-parameter). ``knee`` picks the lam to report:

    x(lam) = improvement of the target metric over lam = 0, scaled to [0, 1] by its largest value
    y(lam) = NDCG@K scaled to [0, 1] between its smallest and largest value over the grid

and the knee is the point furthest above the straight line from the first to the last lam
(the "Kneedle" idea): up to there the target metric improves cheaply, beyond it every further
gain costs disproportionately much accuracy. If no point lies above the line, or the objective
never improves, lam = 0 (no re-ranking is worth its cost).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from project.metrics import METRIC_INFO, EvaluationContext, evaluate
from project.rerankers.greedy import rerank
from project.utils.data_formats import TOPK_FINAL

DEFAULT_LAMBDAS = tuple(np.round(np.linspace(0.0, 1.0, 11), 2))


def sweep(recs: pd.DataFrame, objective, ctx: EvaluationContext, lambdas: Sequence[float] = DEFAULT_LAMBDAS,
          k: int = TOPK_FINAL, name: str = "") -> pd.DataFrame:
    """One row per lam: every metric of the re-ranked top-k lists (only the top k slots are re-ranked)."""
    rows = []
    for lam in lambdas:
        res = evaluate(rerank(recs, objective, lam, ctx, depth=k), ctx, k, name)
        rows.append({"name": name, "objective": objective.name, "lambda": float(lam),
                     **res.summary.drop(["name", "split", "k"]).to_dict()})
    return pd.DataFrame(rows)


def knee(curve: pd.DataFrame, target_metric: str, accuracy: str = "ndcg", k: int = TOPK_FINAL) -> float:
    """lam at the knee of one sweep (``curve`` as returned by ``sweep``, sorted by lam)."""
    curve = curve.sort_values("lambda")
    sign = 1.0 if METRIC_INFO[target_metric][1] else -1.0
    gain = sign * (curve[f"{target_metric}@{k}"].to_numpy() - curve[f"{target_metric}@{k}"].iloc[0])
    acc = curve[f"{accuracy}@{k}"].to_numpy()
    if gain.max() <= 0 or acc.max() == acc.min():
        return 0.0
    x = gain / gain.max()
    y = (acc - acc.min()) / (acc.max() - acc.min())
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    above = dx * (y - y[0]) - dy * (x - x[0])  # > 0: above the line from the first to the last point
    best = int(np.argmax(above))  # first maximum = smallest lam among ties
    return float(curve["lambda"].iloc[best]) if above[best] > 0 else 0.0
