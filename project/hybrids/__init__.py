"""Hybrid recommenders (Task 1).

Track C, the weighted hybrid with regression-learned coefficients (Tasks 1.3, 1.5, 2.3, 2.6):
    weighted          score matrices in; per-user rescaling; regression table; fit_weights; combine;
                      group-aware weights (Task 2.6); cross-fitting over user folds; lists_to_scores (Task 3.3)
    weighted_tuning   the settings grid, cross-validated selection on validation NDCG@10, fit_hybrid
    runnable entry point: python -m project.experiments.hybrid (see project/hybrids/README.md);
    coefficient analysis (Task 2.3): project.analysis.coefficients

Track D, the content-based model and the other hybrids (Tasks 1.1, 1.4, 1.5):
    content_based, switching, mixed, tuning (switching and mixed hybrids)
"""
