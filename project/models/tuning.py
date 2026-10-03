"""Grid search per model with RecBole's ``HyperTuning`` (exhaustive), target = validation NDCG@10.

The search spaces are in ``project/configs/hyper/<Model>.hyper`` (RecBole's format);
the winning configuration is written to ``project/configs/models/<Model>.yaml`` and
all trials to ``results/processed/tuning/<Model>.csv``.
"""
from __future__ import annotations

import json
import logging
from functools import partial
from logging import getLogger
from typing import Dict, List

import pandas as pd
import yaml

from recbole.config import Config
from recbole.trainer import HyperTuning
from recbole.utils import get_trainer, init_logger, init_seed

from project.models.pipeline import prepare_data
from project.models.registry import BASE_CONFIG, TUNING_CONFIG, ModelSpec
from project.utils.data_formats import CHECKPOINTS_DIR, DATASET_NAME
from project.utils.paths import DATASET_DIR, RESULTS_PROCESSED_DIR

TUNING_RESULTS_DIR = RESULTS_PROCESSED_DIR / "tuning"


def _objective(config_dict: dict, config_file_list: List[str], spec: ModelSpec):
    """Like ``recbole.quick_start.objective_function`` but without checkpoints / ray."""
    config = Config(
        model=spec.model_arg(),
        dataset=DATASET_NAME,
        config_file_list=config_file_list,
        config_dict={**config_dict, "data_path": str(DATASET_DIR), "checkpoint_dir": str(CHECKPOINTS_DIR)},
    )
    logger = getLogger()
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    logger.setLevel(logging.WARNING)
    dataset, train_data, valid_data, test_data = prepare_data(config)
    init_seed(config["seed"], config["reproducibility"])
    model = config.model_class(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=False)
    return {
        "model": spec.name,
        "best_valid_score": float(best_valid_score),
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": {k: float(v) for k, v in best_valid_result.items()},
        "test_result": {},  # the test set is only touched by run_models --mode tuned
    }


def tune_model(spec: ModelSpec) -> Dict:
    if not spec.tunable:
        raise ValueError(f"{spec.name} has no search space (baseline)")
    fixed = [str(spec.lecturer_config), str(BASE_CONFIG), str(TUNING_CONFIG)]
    hp = HyperTuning(
        partial(_objective, spec=spec),
        algo="exhaustive",
        early_stop=10_000,  # never stop early: evaluate the whole grid
        params_file=str(spec.hyper_file),
        fixed_config_file_list=fixed,
    )
    print(f"[Track A] tuning {spec.name}: {hp.max_evals} configurations from {spec.hyper_file.name}")
    hp.run()

    rows = []
    for params_str, res in hp.params2result.items():
        params = _parse_params(params_str)
        rows.append({"model": spec.name, **params, **{f"valid_{k}": v for k, v in res["best_valid_result"].items()}})
    trials = pd.DataFrame(rows).sort_values("valid_ndcg@10", ascending=False).reset_index(drop=True)
    TUNING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials.to_csv(TUNING_RESULTS_DIR / f"{spec.name}.csv", index=False)

    best = {k: _plain(v) for k, v in hp.best_params.items()}
    header = (
        f"# Tuned hyper-parameters for {spec.name} -- written by project.experiments.tune_models, do not edit by hand.\n"
        f"# Search space: project/configs/hyper/{spec.name}.hyper ({hp.max_evals} configurations, exhaustive grid).\n"
        f"# Selection: best validation NDCG@10 = {hp.best_score:.4f}. All trials: results/processed/tuning/{spec.name}.csv\n"
    )
    spec.tuned_config.parent.mkdir(parents=True, exist_ok=True)
    spec.tuned_config.write_text(header + yaml.safe_dump(best, sort_keys=True), encoding="utf-8")
    print(f"[Track A] {spec.name}: best {best} -> valid NDCG@10 {hp.best_score:.4f}; wrote {spec.tuned_config}")
    return {"model": spec.name, "n_trials": int(hp.max_evals), "best_params": json.dumps(best),
            "valid_ndcg@10_tuned": float(hp.best_score)}


def _plain(v):
    """Make a searched value YAML-friendly: numpy scalars -> python, and list-valued choices,
    which RecBole's exhaustive search only accepts as strings (``'[64,32]'``), -> real lists."""
    import ast

    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, str):
        try:
            return _plain(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return v
    return v


def _parse_params(params_str: str) -> dict:
    """``HyperTuning.params2str`` joins ``k:v`` pairs with ', '; values are python literals."""
    import ast

    out = {}
    for part in params_str.split(", "):
        k, v = part.split(":", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out
