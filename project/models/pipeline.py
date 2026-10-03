"""Train one model with RecBole, evaluate it, and export its outputs in the shared formats.

Mirrors ``recbole.quick_start.run_recbole`` (same seeding, same data preparation, same
trainer) but keeps the model object, so we can (1) dump the full score matrix for every
user-item pair, (2) write the top-50 candidate lists, and (3) check that the export
reproduces RecBole's own Precision@10 / Recall@10 for the same checkpoint.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from logging import getLogger
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_trainer, init_logger, init_seed

from project.models.registry import ModelSpec
from project.utils.data_formats import (
    CHECKPOINTS_DIR,
    DATASET_NAME,
    EVAL_SPLITS,
    RECOMMENDATION_COLUMNS,
    TOPK_CANDIDATES,
    TOPK_FINAL,
    ScoreMatrix,
    load_ground_truth,
    precision_recall_at_k,
    save_recommendations,
    save_scores,
)
from project.utils.paths import DATASET_DIR

METRIC_COLUMNS = ["ndcg@10", "recall@10", "precision@10", "mrr@10", "hit@10", "map@10"]


def build_config(spec: ModelSpec, mode: str, overrides: Optional[dict] = None) -> Config:
    config_dict = {
        "data_path": str(DATASET_DIR),  # RecBole appends the dataset name itself
        "checkpoint_dir": str(CHECKPOINTS_DIR),
        **(overrides or {}),
    }
    return Config(
        model=spec.model_arg(),
        dataset=DATASET_NAME,
        config_file_list=spec.config_files(mode),
        config_dict=config_dict,
    )


def prepare_data(config: Config):
    """Seed, build the dataset and the (frozen) train/valid/test dataloaders."""
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    return dataset, train_data, valid_data, test_data


def _reset_logger(config: Config):
    logger = getLogger()
    for h in logger.handlers[:]:  # avoid duplicated lines when several models run in one process
        logger.removeHandler(h)
    init_logger(config)
    logging.getLogger("ray").setLevel(logging.ERROR)


@dataclass
class RunResult:
    name: str
    mode: str
    best_valid: Dict[str, float]
    test: Dict[str, float]
    best_epoch: int
    train_seconds: float
    config_files: list = field(default_factory=list)
    hyper_parameters: dict = field(default_factory=dict)


def train_and_evaluate(spec: ModelSpec, mode: str, overrides: Optional[dict] = None):
    """Train ``spec`` and return ``(RunResult, model, dataset, valid_data, test_data, config)``."""
    config = build_config(spec, mode, overrides)
    _reset_logger(config)
    logger = getLogger()
    logger.info(f"[Track A] {spec.name} ({mode}) config files: {spec.config_files(mode)}")

    dataset, train_data, valid_data, test_data = prepare_data(config)
    init_seed(config["seed"], config["reproducibility"])
    model = config.model_class(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.saved_model_file = str(CHECKPOINTS_DIR / f"{spec.name}_{mode}.pth")

    t0 = time.time()
    _, best_valid = trainer.fit(train_data, valid_data, saved=True, show_progress=False, verbose=True)
    train_seconds = time.time() - t0
    test = trainer.evaluate(test_data, load_best_model=True, show_progress=False)  # loads best epoch
    best_epoch = int(torch.load(trainer.saved_model_file, map_location="cpu", weights_only=False)["epoch"])
    logger.info(f"[Track A] {spec.name} best valid {dict(best_valid)}")
    logger.info(f"[Track A] {spec.name} test {dict(test)}")

    result = RunResult(
        name=spec.name,
        mode=mode,
        best_valid={k: float(v) for k, v in best_valid.items()},
        test={k: float(v) for k, v in test.items()},
        best_epoch=best_epoch,
        train_seconds=train_seconds,
        config_files=spec.config_files(mode),
        hyper_parameters=_hyper_parameters(spec, config),
    )
    return result, model, dataset, valid_data, test_data, config


def _hyper_parameters(spec: ModelSpec, config: Config) -> dict:
    """The model's own hyper-parameters (keys of the lecturer's config / the search space)."""
    import yaml

    keys = set()
    for f in [spec.lecturer_config] + ([spec.tuned_config] if spec.tuned_config.exists() else []):
        with open(f, encoding="utf-8") as fh:
            keys |= set((yaml.safe_load(fh) or {}).keys())
    skip = {"model", "dataset", "data_path", "eval_args", "metrics", "topk", "valid_metric",
            "show_progress", "seed", "reproducibility", "checkpoint_dir"}
    return {k: config[k] for k in sorted(keys - skip) if k in config.final_config_dict}


# ---------------------------------------------------------------------------- export

@torch.no_grad()
def _full_sort_scores(model, dataset, interaction, batch_size):
    """``full_sort_predict`` or, for models without it (NeuMF), the same pointwise
    fallback RecBole's trainer uses: every user repeated for every item, ``predict``
    in chunks of ``eval_batch_size`` rows."""
    try:
        return model.full_sort_predict(interaction)
    except NotImplementedError:
        n_items = dataset.item_num
        item_feat = dataset.get_item_feature().to(interaction.device if hasattr(interaction, "device") else "cpu")
        new_inter = interaction.repeat_interleave(n_items)
        new_inter.update(item_feat.repeat(len(interaction)))
        total = len(new_inter)
        if total <= batch_size:
            return model.predict(new_inter)
        out = []
        for start in range(0, total, batch_size):
            out.append(model.predict(new_inter[start:min(start + batch_size, total)]))
        return torch.cat(out)

@torch.no_grad()
def scores_and_topk(model, dataset, eval_data, config, k: int = TOPK_CANDIDATES):
    """Scores of all items for every user of ``eval_data`` and their top-k lists.

    Same computation as RecBole's evaluation (``Trainer._full_sort_batch_eval``): one
    ``full_sort_predict`` per user batch, ``[pad]`` and history items set to ``-inf``,
    then ``torch.topk(k=10)`` on that very tensor (same dtype, same batch, same k), so
    ranks 1-10 of the exported lists are exactly the lists RecBole scored -- including
    how ties are broken, which matters for Pop and the kNN models; ranks 11-k are the
    next best items. The stored matrix is float32 with NaN for history items.
    """
    model.eval()
    uid_field, iid_field = dataset.uid_field, dataset.iid_field
    uid_list = eval_data.uid_list.numpy()
    batch = int(config["eval_batch_size"])
    matrices, rows = [], []
    for start in range(0, len(uid_list), batch):
        uids = uid_list[start:start + batch]
        interaction = dataset.join(Interaction({uid_field: torch.tensor(uids)})).to(config["device"])
        scores = _full_sort_scores(model, dataset, interaction, batch).view(-1, dataset.item_num)
        scores[:, 0] = -np.inf
        for row, uid in enumerate(uids):
            hist = eval_data.uid2history_item[uid]
            if hist is not None and len(hist):
                scores[row, hist] = -np.inf
        # torch.topk breaks ties differently for different k, so take the final list of
        # TOPK_FINAL exactly as RecBole does (same k), then the next k - TOPK_FINAL items.
        m = scores.float().cpu().numpy()  # stored matrix (before the top-10 is blanked out)
        head_scores, head_items = torch.topk(scores, TOPK_FINAL, dim=1)
        scores.scatter_(1, head_items, -np.inf)
        tail_scores, tail_items = torch.topk(scores, k - TOPK_FINAL, dim=1)
        top_scores = torch.cat([head_scores, tail_scores], dim=1).cpu().numpy()
        top_items = torch.cat([head_items, tail_items], dim=1).cpu().numpy()
        user_tokens = dataset.id2token(uid_field, uids).astype(str)
        for row, user in enumerate(user_tokens):
            item_tokens = dataset.id2token(iid_field, top_items[row]).astype(str)
            for rank in range(k):
                if not np.isfinite(top_scores[row, rank]):
                    break
                rows.append((user, item_tokens[rank], rank + 1, float(top_scores[row, rank])))
        m[~np.isfinite(m)] = np.nan
        matrices.append(m[:, 1:])  # drop the [pad] column
    sm = ScoreMatrix(
        dataset.id2token(uid_field, uid_list).astype(str),
        dataset.id2token(iid_field, np.arange(1, dataset.item_num)).astype(str),
        np.concatenate(matrices, axis=0).astype(np.float32),
    )
    return sm, pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def export_outputs(spec: ModelSpec, model, dataset, valid_data, test_data, config, test_result=None) -> dict:
    """Write ``<name>_{valid,test}.npz`` and ``<name>_{valid,test}_top50.csv``; verify on test."""
    paths = {}
    for split, loader in zip(EVAL_SPLITS, (valid_data, test_data)):
        sm, recs = scores_and_topk(model, dataset, loader, config, TOPK_CANDIDATES)
        paths[f"scores_{split}"] = save_scores(sm, spec.name, split)
        paths[f"recs_{split}"] = save_recommendations(recs, spec.name, split, TOPK_CANDIDATES)
        if split == "test" and test_result is not None:
            _verify_export(spec, recs, test_result)
    return paths


def _verify_export(spec: ModelSpec, recs: pd.DataFrame, test_result: Dict[str, float], tol: float = 2.5e-4):
    """Our exported top-10 must reproduce RecBole's Precision@10 and Recall@10 (4 decimals)."""
    p, r = precision_recall_at_k(recs, load_ground_truth("test"), TOPK_FINAL)
    dp, dr = abs(p - test_result[f"precision@{TOPK_FINAL}"]), abs(r - test_result[f"recall@{TOPK_FINAL}"])
    msg = (f"[Track A] {spec.name} export check: precision@10 {p:.4f} (RecBole {test_result['precision@10']:.4f}), "
           f"recall@10 {r:.4f} (RecBole {test_result['recall@10']:.4f})")
    getLogger().info(msg)
    if spec.verify_export and (dp > tol or dr > tol):
        raise RuntimeError("exported recommendation lists do not reproduce RecBole's metrics -- " + msg)
