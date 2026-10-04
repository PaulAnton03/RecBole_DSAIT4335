"""Which individual models Track A runs, and where their configs live.

Config precedence (later files override earlier ones, RecBole semantics):

    recbole/config/<Model>/ml-100k.yaml   lecturer's model config (hyper-parameters, UNTUNED)
    project/configs/base.yaml             our shared evaluation protocol (split, NDCG@10, ...)
    project/configs/tuning.yaml           training schedule for the search / tuned runs   (mode=tuned)
    project/configs/models/<Model>.yaml   tuned hyper-parameters written by tune_models    (mode=tuned)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type

from project.utils.paths import PROJECT_CONFIGS_DIR, REPO_ROOT

LECTURER_CONFIG_DIR = REPO_ROOT / "recbole" / "config"
BASE_CONFIG = PROJECT_CONFIGS_DIR / "base.yaml"
TUNING_CONFIG = PROJECT_CONFIGS_DIR / "tuning.yaml"
HYPER_DIR = PROJECT_CONFIGS_DIR / "hyper"
TUNED_DIR = PROJECT_CONFIGS_DIR / "models"

MODES = ("quick", "tuned")


@dataclass(frozen=True)
class ModelSpec:
    name: str  # our name, used in every file name
    recbole_model: str  # RecBole class name (UserKNN is RecBole's ItemKNN with knn_method=user)
    family: str
    description: str
    baseline: bool = False
    tunable: bool = True  # has a search space in project/configs/hyper/<name>.hyper
    model_class: Optional[Type] = None  # custom class (registered instead of recbole_model)
    verify_export: bool = True  # exported top-10 must reproduce RecBole's Precision/Recall@10

    @property
    def lecturer_config(self) -> Path:
        return LECTURER_CONFIG_DIR / self.name / "ml-100k.yaml"

    @property
    def hyper_file(self) -> Path:
        return HYPER_DIR / f"{self.name}.hyper"

    @property
    def tuned_config(self) -> Path:
        return TUNED_DIR / f"{self.name}.yaml"

    def config_files(self, mode: str) -> List[str]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        files = [self.lecturer_config, BASE_CONFIG]
        if mode == "tuned" and self.tunable:
            if not self.tuned_config.exists():
                raise FileNotFoundError(
                    f"{self.tuned_config} not found -- run `python -m project.experiments.tune_models "
                    f"--models {self.name}` first (or use --mode quick)."
                )
            files += [TUNING_CONFIG, self.tuned_config]
        return [str(f) for f in files]

    def model_arg(self):
        """What to hand to ``recbole.config.Config(model=...)``."""
        return self.model_class if self.model_class is not None else self.recbole_model


def _random_class():
    from project.models.random_per_user import RandomPerUser

    return RandomPerUser


def _pop_class():
    from project.models.pop_exact import PopExact

    return PopExact


_SPECS = [
    ModelSpec("Random", "Random", "baseline", "uniform random scores (per user)", baseline=True,
              tunable=False, model_class=_random_class()),
    ModelSpec("Pop", "Pop", "baseline", "most-popular items in the training set (exact counts)", baseline=True,
              tunable=False, model_class=_pop_class()),
    ModelSpec("ItemKNN", "ItemKNN", "neighbourhood", "item-based cosine kNN (k, shrink)"),
    ModelSpec("UserKNN", "ItemKNN", "neighbourhood", "user-based cosine kNN (RecBole ItemKNN, knn_method=user)"),
    ModelSpec("EASE", "EASE", "linear", "closed-form item-item linear model (reg_weight)"),
    ModelSpec("SLIMElastic", "SLIMElastic", "linear", "sparse linear item model, ElasticNet (alpha, l1_ratio)"),
    ModelSpec("BPR", "BPR", "latent factor", "matrix factorisation with BPR pairwise loss"),
    ModelSpec("NeuMF", "NeuMF", "neural", "neural collaborative filtering (GMF + MLP)"),
    ModelSpec("FISM", "FISM", "neural", "factored item similarity model"),
    ModelSpec("LightGCN", "LightGCN", "graph", "light graph convolution over the user-item graph"),
    ModelSpec("NGCF", "NGCF", "graph", "neural graph collaborative filtering"),
]
MODELS = {s.name: s for s in _SPECS}

# The plan's shortlist (one model per lecture family + the two baselines); SLIMElastic,
# FISM and NGCF are near-duplicates of EASE / NeuMF / LightGCN and are run with --models all.
DEFAULT_MODELS = ["Random", "Pop", "ItemKNN", "UserKNN", "EASE", "BPR", "NeuMF", "LightGCN"]
ALL_MODELS = list(MODELS)


def resolve_models(arg: str) -> List[ModelSpec]:
    """``"default"``, ``"all"`` or a comma-separated list of names."""
    if arg == "default":
        names = DEFAULT_MODELS
    elif arg == "all":
        names = ALL_MODELS
    else:
        names = [n.strip() for n in arg.split(",") if n.strip()]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {ALL_MODELS}")
    return [MODELS[n] for n in names]
