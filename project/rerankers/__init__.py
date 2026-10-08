"""Re-rankers: diversification, calibration, fairness (Task 3, Track E).

One greedy loop (``greedy.py``) re-orders a model's top-50 candidates per user, trading the model's
relevance against one beyond-accuracy objective (``objectives.py``) with a single lam in [0, 1]:

    from project.metrics import EvaluationContext
    from project.rerankers import MMR, rerank
    from project.utils.data_formats import load_recommendations

    ctx = EvaluationContext.for_split("valid")
    recs = rerank(load_recommendations("EASE", "valid"), MMR(), lam=0.3, ctx=ctx)   # shared format, 50 per user

``tradeoff.py`` sweeps lam and picks the knee; ``python -m project.experiments.rerank_models`` runs the
whole Task 3.2 experiment. See ``project/rerankers/README.md``.
"""
from project.rerankers.greedy import greedy_order, list_reranker, normalise_scores, rerank, reranked_name
from project.rerankers.objectives import MMR, OBJECTIVES, Calibrated, Objective, PopCalibrated, XQuAD, make_objective
from project.rerankers.tradeoff import DEFAULT_LAMBDAS, knee, sweep

__all__ = ["rerank", "list_reranker", "greedy_order", "normalise_scores", "reranked_name", "Objective", "MMR",
           "Calibrated", "PopCalibrated", "XQuAD", "OBJECTIVES", "make_objective", "sweep", "knee", "DEFAULT_LAMBDAS"]
