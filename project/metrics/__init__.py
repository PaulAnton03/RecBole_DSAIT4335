"""Evaluation metrics, accuracy and beyond-accuracy (Task 2.1, Track B).

Our own implementation, independent of RecBole's evaluator (RecBole only produces the lists).
Start with ``evaluate`` (see ``evaluate.py``) and ``project/metrics/README.md`` for every formula
and convention. The building blocks are importable on their own, e.g. for the re-rankers:

    accuracy      Precision, Recall, F1, Hit, NDCG, MRR, MAP
    diversity     intra-list diversity (genre Jaccard), catalogue coverage
    novelty       self-information novelty, serendipity (relevant and not in the popularity list)
    calibration   genre miscalibration KL(p || q~)
    popularity    average popularity, long-tail share, user popularity deviation (UPD)
    fairness      GRU, item exposure, Gini, entropy, group exposure
    lookups       genres, popularity, head/mid/tail item groups, user activity groups
"""
from project.metrics.evaluate import (METRIC_INFO, EvaluationContext, EvaluationResult, evaluate,
                                      evaluate_saved)

__all__ = ["EvaluationContext", "EvaluationResult", "evaluate", "evaluate_saved", "METRIC_INFO"]
