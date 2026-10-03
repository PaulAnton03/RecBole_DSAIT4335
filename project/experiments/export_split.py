"""Create the ONE frozen train/validation/test split everybody uses (Track A, day 1).

Writes ``results/raw/splits/ml-100k.{train,valid,test}.tsv`` (external ids, with rating
and timestamp) and records their SHA-256 checksums in
``results/processed/split_manifest.json`` (tracked in git). Re-running verifies the
regenerated files against the manifest, so every machine provably works on the same
split; ``--check-all-models`` additionally rebuilds the split through every model's
config and asserts it is identical (RecBole splits before the model is built, so only
``seed`` and ``eval_args`` matter -- both are fixed in ``project/configs/base.yaml``).

Usage (repository root)::

    python -m project.experiments.export_split                 # write + verify / create manifest
    python -m project.experiments.export_split --check-all-models
"""
from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from project.models.pipeline import build_config, prepare_data
from project.models.registry import MODELS, ALL_MODELS
from project.utils.data_formats import DATASET_NAME, SPLIT_COLUMNS, SPLIT_PARTS, save_split, split_path
from project.utils.paths import DATASET_DIR, RESULTS_PROCESSED_DIR

MANIFEST = RESULTS_PROCESSED_DIR / "split_manifest.json"


def split_frames(model_name: str = "Pop") -> dict:
    config = build_config(MODELS[model_name], "quick")
    dataset, train_data, valid_data, test_data = prepare_data(config)
    # RecBole min-max normalises float fields in memory (rating 1..5 -> 0..1, timestamp -> 0), so the
    # original rating and timestamp are taken from the raw dataset file instead.
    raw = pd.read_csv(DATASET_DIR / DATASET_NAME / f"{DATASET_NAME}.inter", sep="\t",
                      dtype={"user_id:token": str, "item_id:token": str})
    raw.columns = [c.split(":")[0] for c in raw.columns]
    raw = raw.drop_duplicates(["user_id", "item_id"])
    frames = {}
    for part, loader in zip(SPLIT_PARTS, (train_data, valid_data, test_data)):
        inter = loader.dataset.inter_feat
        df = pd.DataFrame({
            "user_id": dataset.id2token(dataset.uid_field, inter[dataset.uid_field].numpy()).astype(str),
            "item_id": dataset.id2token(dataset.iid_field, inter[dataset.iid_field].numpy()).astype(str),
        })
        df = df.merge(raw[SPLIT_COLUMNS], on=["user_id", "item_id"], how="left", validate="one_to_one")
        assert df["rating"].notna().all(), "every split interaction must exist in the raw .inter file"
        df["rating"] = df["rating"].astype(int)
        df["timestamp"] = df["timestamp"].astype("int64")
        frames[part] = df[SPLIT_COLUMNS]
    return frames


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-all-models", action="store_true",
                        help="rebuild the split through every model config and assert it is identical")
    parser.add_argument("--update-manifest", action="store_true",
                        help="overwrite results/processed/split_manifest.json (only when the protocol changes!)")
    args = parser.parse_args()

    frames = split_frames("Pop")
    stats = {}
    for part, df in frames.items():
        path = save_split(df, part)
        stats[part] = {"interactions": int(len(df)), "users": int(df.user_id.nunique()),
                       "items": int(df.item_id.nunique()), "sha256": _sha256(path)}
        print(f"wrote {path}: {len(df)} interactions, {df.user_id.nunique()} users, {df.item_id.nunique()} items")

    if MANIFEST.exists() and not args.update_manifest:
        manifest = json.loads(MANIFEST.read_text())
        for part in SPLIT_PARTS:
            if manifest[part]["sha256"] != stats[part]["sha256"]:
                raise SystemExit(f"{split_path(part)} differs from the frozen split in {MANIFEST} -- "
                                 "do NOT change seed/eval_args; if the group agreed on a new protocol, "
                                 "re-run with --update-manifest and tell everybody.")
        print(f"split matches the frozen manifest {MANIFEST}")
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({"dataset": "ml-100k", "seed": 2020, "eval_args": "RS [0.8,0.1,0.1], group_by user, order RO, mode full",
                                        **stats}, indent=2) + "\n")
        print(f"wrote manifest {MANIFEST}")

    if args.check_all_models:
        for name in ALL_MODELS:
            other = split_frames(name)
            for part in SPLIT_PARTS:
                if not other[part].equals(frames[part]):
                    raise SystemExit(f"split built through {name}'s config differs in {part}!")
            print(f"{name:12s} builds the identical split")


if __name__ == "__main__":
    main()
