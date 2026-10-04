# Track A — individual models (Tasks 1.1, 1.2, 2.2)

Owner: Person A ("Models"). This folder trains, tunes and exports the individual recommenders every other track
builds on. Nothing here is edited by other tracks; they only **read** the files described in §2.

Everything is run from the repository root inside the virtual environment (see `PROJECT.md` §3).

## 1. One command per step

| Step | Command | Output |
| --- | --- | --- |
| 0. Frozen split (day 1, once) | `python -m project.experiments.export_split` | `results/raw/splits/ml-100k.{train,valid,test}.tsv`, checksums in `results/processed/split_manifest.json` |
| 1. Quick, untuned scores (day 2) | `python -m project.experiments.run_models --mode quick --models all` | score files + top-50 lists for all 11 models, `results/processed/recbole_metrics_quick.csv` |
| 2. Tuning (Task 1.2) | `python -m project.experiments.tune_models` | `project/configs/models/<Model>.yaml`, `results/processed/tuning/<Model>.csv`, `results/processed/tuning_summary.csv` |
| 3. Final, tuned scores (release) | `python -m project.experiments.run_models --mode tuned` | score files + top-50 lists of the tuned models, `results/processed/recbole_metrics_tuned.csv` |
| 4. Our metrics on every list (Track B) | `python -m project.experiments.evaluate_models` | `results/processed/metrics_{valid,test}.csv` (see `project/metrics/README.md`) |
| 5. Main results table (Task 2.2) | `python -m project.experiments.results_table` | `report/tables/generated/{model_results,tuning_summary}.tex`, `figures/generated/model_comparison.pdf`, `results/processed/model_results.csv` |
| Tests | `python -m pytest project/tests -q` | |

`--models` accepts `default` (the plan's shortlist: Random, Pop, ItemKNN, UserKNN, EASE, BPR, NeuMF, LightGCN),
`all` (adds SLIMElastic, FISM, NGCF) or a comma-separated list. Smoke test: `--epochs 2` (writes to a `*_smoke.csv`).
Runtimes on an 8-core laptop CPU: step 1 ≈ 3 min, step 2 ≈ 30–60 min, step 3 ≈ 5 min.

Re-running a step overwrites its outputs in place (deterministic names, seed 2020), so a re-run gives a clean diff.
`results/raw/` is git-ignored; after the **release** of the final scores (step 3) the raw folder is zipped and shared
in the group chat, and nobody retrains.

## 2. Shared formats (agreed on day 1)

All ids are the original MovieLens ids as strings; RecBole's internal ids never leave this folder.
Loader/writer helpers for every format are in `project/utils/data_formats.py`.

| File | Columns / arrays | Written by | Read by |
| --- | --- | --- | --- |
| `results/raw/splits/ml-100k.{train,valid,test}.tsv` | `user_id, item_id, rating, timestamp` | `export_split` | everybody (`load_split`, `load_ground_truth`, `load_history`) |
| `results/raw/scores/<Model>_{valid,test}.npz` | `user_id (U,)`, `item_id (I,)`, `score (U×I) float32`; **NaN = item in the user's history** | `run_models` | hybrids (C, D), re-rankers (E) via `load_scores` → `ScoreMatrix` (`.to_frame()` for long format) |
| `results/raw/recommendations/<Name>_{valid,test}_top50.csv` | `user_id, item_id, rank, score` (rank 1 = best, 50 rows per user) | every model, hybrid and re-ranker (`save_recommendations`) | metrics (B) via `load_recommendations`; the final list is `rank <= 10` |

Protocol fixed in `project/configs/base.yaml`: seed 2020, RecBole random split 80/10/10 per user (`RS`, `RO`,
`group_by: user`), full ranking over all unseen items, relevance = every held-out interaction (no rating threshold;
the rating is in the split files if the group decides otherwise), final list 10, candidate list 50, model
selection on **validation NDCG@10**. For `valid` the masked history is the train set, for `test` it is train + valid —
exactly what RecBole masks.

**Guarantees** (checked at run time): the split is byte-identical to the manifest and identical through every
model's config (`export_split --check-all-models`); ranks 1–10 of every exported list reproduce RecBole's own
Precision@10 / Recall@10 for the same checkpoint to 4 decimals (the run aborts otherwise). This means Track B's
metric implementations can be validated against RecBole's numbers on these very lists.

## 3. Models and configuration

| Name | RecBole class | Family | Tuned hyper-parameters (`project/configs/hyper/<Name>.hyper`) |
| --- | --- | --- | --- |
| Random | `Random` (our per-user variant, `random_per_user.py`) | baseline | – |
| Pop | `Pop` (our exact-count variant, `pop_exact.py`) | baseline | – |
| ItemKNN | `ItemKNN` (`knn_method: item`) | neighbourhood | `k`, `shrink` |
| UserKNN | `ItemKNN` (`knn_method: user`) | neighbourhood | `k`, `shrink` |
| EASE | `EASE` | linear item–item | `reg_weight` |
| SLIMElastic | `SLIMElastic` | linear item–item | `alpha`, `l1_ratio` |
| BPR | `BPR` | latent factors (pairwise) | `embedding_size`, `learning_rate` |
| NeuMF | `NeuMF` | neural CF | `learning_rate`, `mlp_hidden_size`, `dropout_prob` |
| FISM | `FISM` | neural item-based | `embedding_size`, `learning_rate`, `reg_weights` |
| LightGCN | `LightGCN` | graph | `n_layers`, `learning_rate`, `reg_weight` |
| NGCF | `NGCF` | graph | `learning_rate`, `reg_weight`, `message_dropout` |

Config precedence (later overrides earlier): lecturer's `recbole/config/<Model>/ml-100k.yaml` → `project/configs/base.yaml`
(evaluation protocol) → `project/configs/tuning.yaml` (up to 100 epochs, validate every epoch, early stop after 10) →
`project/configs/models/<Model>.yaml` (tuned values, written by `tune_models`). `--mode quick` stops after `base.yaml`,
i.e. it is the lecturer's hyper-parameters with our evaluation protocol.

Why a custom Random: RecBole's `Random.full_sort_predict` draws one random vector per *batch* and repeats it for
every user, so with one evaluation batch all users would get the same list. Ours draws per user (seeded), which is
what a random baseline should be for coverage/diversity metrics.

Why a custom Pop: RecBole's `Pop` counts with `item_cnt[item] = item_cnt[item] + 1` per training batch, which adds 1
only once per batch for a repeated id, so it counts *batches* containing an item, not interactions. The counts of all
popular films saturate (11 distinct scores on ML-100K) and `torch.topk`'s tie-breaking picks the list (test NDCG@10
0.100–0.103 depending on the machine). Ours counts every training interaction once (test NDCG@10 0.142).

Reported numbers: every table and figure is computed with our own metrics (`project/metrics`, Task 2.1) on the
exported lists; RecBole's printed metrics (`recbole_metrics_*.csv`) are only used for model selection during tuning
and as a cross-check (`results/processed/metric_validation.csv`).

## 4. Files

```
project/models/registry.py          model list, config file resolution
project/models/pipeline.py          train + evaluate + export (score matrix, top-50, export check)
project/models/tuning.py            grid search with RecBole HyperTuning on validation NDCG@10
project/models/random_per_user.py   per-user Random baseline
project/models/pop_exact.py         Pop baseline with exact interaction counts
project/experiments/export_split.py / run_models.py / tune_models.py / results_table.py
project/utils/data_formats.py       the shared formats (loaders and writers)
project/tests/test_data_formats.py  unit tests
```
