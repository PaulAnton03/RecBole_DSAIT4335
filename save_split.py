"""
save_split.py

Builds the train/valid/test split for a dataset exactly as RecBole would
for a given model + config, and dumps each partition to a plain TSV file
using the ORIGINAL (external) user/item tokens -- not RecBole's internal
integer ids -- so the files can be read by pandas/Excel or by any other
tool outside RecBole.

The random number generators are seeded with the config's `seed` before
splitting, exactly like run_recbole.py does, so the exported split is the
one the model was trained on -- as long as you pass the same config file(s)
you trained with.

Usage:
    python save_split.py --model BPR --dataset ml-100k \
        --config_files recbole/config/BPR/ml-100k.yaml \
        --output_dir saved/splits
"""
import argparse
import os

import pandas as pd

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


def dump_interactions(data_loader, dataset, out_path):
    """Write one partition (train/valid/test) of a dataloader to disk.

    The dataloader's underlying Interaction tensors store *internal* ids.
    We map them back to the original raw ids with dataset.id2token(...)
    so the exported file is human-readable and reproducible outside RecBole.
    """
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    inter_feat = data_loader.dataset.inter_feat
    user_ids = inter_feat[uid_field].numpy()
    item_ids = inter_feat[iid_field].numpy()

    user_tokens = dataset.id2token(uid_field, user_ids)
    item_tokens = dataset.id2token(iid_field, item_ids)

    df = pd.DataFrame({uid_field: user_tokens, iid_field: item_tokens})
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(df)} interactions -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BPR")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config_files", type=str, default=None,
                         help="space separated list of yaml files")
    parser.add_argument("--output_dir", type=str, default="saved/splits")
    args = parser.parse_args()

    config_file_list = args.config_files.split(" ") if args.config_files else None
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
    )
    # same seeding as run_recbole(): without it, order: RO shuffles differently on every run
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    dump_interactions(train_data, dataset, os.path.join(args.output_dir, f"{args.dataset}.train.tsv"))
    dump_interactions(valid_data, dataset, os.path.join(args.output_dir, f"{args.dataset}.valid.tsv"))
    dump_interactions(test_data, dataset, os.path.join(args.output_dir, f"{args.dataset}.test.tsv"))


if __name__ == "__main__":
    main()
