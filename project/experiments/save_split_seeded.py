"""Seeded wrapper around the lecturer's ``save_split.py``.

``save_split.py`` builds the train/valid/test split without calling ``init_seed``, so
its split is random and does NOT match the split used when a model was trained
(``run_recbole.py``) or reloaded (``save_recommendations.py``); scoring against it
gives meaningless numbers. This wrapper seeds RecBole exactly like ``run_recbole.py``
(``seed`` / ``reproducibility`` from the config) and then delegates to the lecturer's
script, which stays untouched.

Usage (from the repository root; same arguments as ``save_split.py``)::

    python -m project.experiments.save_split_seeded --model BPR --dataset ml-100k \
        --config_files recbole/config/BPR/ml-100k.yaml --output_dir results/raw/splits
"""
import argparse

import save_split  # lecturer's script, importable because the repository root is on sys.path
from recbole.config import Config
from recbole.utils import init_seed


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=str, default="BPR")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config_files", type=str, default=None)
    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.split(" ") if args.config_files else None
    config = Config(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])

    save_split.main()


if __name__ == "__main__":
    main()
