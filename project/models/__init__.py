"""Track A -- individual recommendation models (Tasks 1.1, 1.2 and the main results table of 2.2).

* ``registry``  -- which models we run, which config files they use
* ``pipeline``  -- train with RecBole, evaluate, export score matrices + top-50 lists
* ``tuning``    -- grid search on validation NDCG@10 (RecBole ``HyperTuning``)

Entry points: ``python -m project.experiments.{export_split,run_models,tune_models,results_table}``.
"""
