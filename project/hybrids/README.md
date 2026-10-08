# Track C — weighted hybrid (Tasks 1.3, 1.5, 2.3, 2.6)

Owner: Person C ("Weighted hybrid"). Builds the hybrid with regression-learned coefficients, tunes it,
analyses what every member contributes, and derives a group-aware hybrid from Track B's group analysis.
Nothing here calls RecBole: the inputs are Track A's released score matrices and lists in `results/raw/`
(see `project/models/README.md` §2), the yardstick is Track B's `project.metrics`.

## 1. Run

```bash
python -m project.experiments.hybrid                      # tune, fit, export, analyse, write assets (~5–10 min)
python -m project.experiments.hybrid --grid quick         # smoke test (small grid)
python -m project.experiments.hybrid --setting zscore,ridge,0.001,union   # skip tuning, use this setting
python -m project.experiments.hybrid --members ItemKNN,EASE,BPR,NeuMF,LightGCN,ContentBased   # add Track D's model
python -m pytest project/tests/test_hybrids.py -q
```

Afterwards re-run Track B's `evaluate_models` and `group_analysis`: they pick the exported hybrids up by name.

## 2. The model (Task 1.3)

`score'(u, i) = Σ_k w_k · z_k(u, i)`

* `z_k`: member k's score for the pair, **rescaled per user** (`zscore`, `minmax`, `rank` or `none`), because EASE
  weights, BPR dot products, kNN similarity sums and Pop counts live on different scales.
* `w`: coefficients of a **regression model fitted on the validation split only** — one row per candidate
  (user, movie) pair, the members' rescaled scores as features, target 1 if the movie is a held-out validation
  interaction. Candidate rows: `union` of the members' top-50 validation lists (the region that decides the
  final top-10, with the members' hard negatives) or `all` unseen movies. Regressions: `ridge` (closed form, the
  literal reading of the lecture's weighted sum), `nnls` (non-negative weights) and `logistic`.
  Training interactions are never used (the members have memorised them); the test split is touched once.
* Members: the plan's shortlist, one model per lecture family — ItemKNN, EASE, BPR, NeuMF, LightGCN (plus the
  content-based model once Track D exports it as `ContentBased_{valid,test}.npz`). "All models" is an ablation:
  near-duplicates (EASE/SLIM, ItemKNN/UserKNN, BPR/NeuMF/LightGCN) split the credit and blur the analysis.

Sanity check built into the tests: weight 1 on one member reproduces that member's ranking exactly, so a learned
hybrid cannot be worse than its best member on the data it was fitted on.

## 3. Tuning (Task 1.5)

The weights are learned on validation, so settings are compared by **5-fold cross-fitting over users on the
validation split**: every user's hybrid list comes from weights that never saw the user's validation
interactions (`project/hybrids/weighted_tuning.py`). Grid: rescaling × regression × regularisation × candidate rule
(30 settings); selection on NDCG@10. The selected setting is refitted on all validation users and evaluated once
on test. Because the exported *validation* lists of the hybrids are in-sample (weights fitted on the same users),
`hybrid_tuning.csv` / `hybrid_results.csv` also give the cross-validated validation number, which is the one to
quote.

## 4. Outputs

| File | Content |
| --- | --- |
| `results/raw/scores/<Hybrid>_{valid,test}.npz`, `results/raw/recommendations/<Hybrid>_{valid,test}_top50.csv` | `WeightedHybrid` (shortlist), `WeightedHybridAll` (all models), `GroupHybrid` (Task 2.6) in the shared formats |
| `results/processed/hybrid_tuning.csv` | every setting: cross-validated NDCG@10 / Recall@10 / Precision@10, rows, positives |
| `results/processed/hybrid_weights.csv` | learned weights: `coef` (feature scale, what `combine` uses), `coef_std` (per standard deviation of the feature), `share` (normalised to sum 1), 95 % bootstrap interval over users |
| `results/processed/hybrid_ablation.csv` | full model, leave-one-member-out, every single member, all models, + Pop: cross-validated and test NDCG@10 with the difference to the full model |
| `results/processed/hybrid_attribution.csv`, `..._counts.csv` | hit attribution (Task 2.3): per member, the hybrid's test hits whose largest weighted term is that member, by popularity group; and how many members had each hit in their own top-10 |
| `results/processed/hybrid_member_overlap.csv` | per member pair: correlation of rescaled scores on the candidate rows, mean Jaccard overlap of top-10 lists |
| `results/processed/hybrid_groups.csv`, `hybrid_group_weights.csv` | group-aware hybrid vs. global hybrid per user group (paired bootstrap), per-group weights |
| `results/processed/hybrid_results.csv` | summary: best member vs. the hybrids on validation (cross-validated and in-sample) and test |
| `results/processed/hybrid_significance.csv` | paired per-user NDCG@10 differences (hybrid vs. best member; group-aware vs. global) with the 95 % bootstrap interval of Task 2.5 and a Wilcoxon signed-rank test; validation uses the cross-fitted lists |
| `figures/generated/hybrid_coefficients.pdf`, `report/tables/generated/{hybrid_results,hybrid_ablation,hybrid_groups}.tex` | report assets |

## 5. Coefficient analysis (Task 2.3) and the group-aware hybrid (Task 2.6)

Weights alone mislead when members are similar, so `project/analysis/coefficients.py` adds: bootstrap intervals
(resample users, refit), leave-one-member-out drops (the member's real contribution), hit attribution (for every
correct movie in the hybrid's top-10, the member whose weighted term is largest, and whether any member had the
movie in its own top-10), and the pairwise score correlation / top-10 overlap of the members.

The group-aware hybrid fits the same regression separately per user group of one dimension of Task 2.5
(`activity`: low / medium / high training interactions; `taste`: niche / mixed / mainstream). The dimension is
chosen by the cross-validated validation NDCG@10; the comparison with the global hybrid is made per group with
paired bootstrap intervals on test (`hybrid_groups.csv`).

## 6. Files

```
project/hybrids/weighted.py        aligned_scores, rescale, candidate_mask, regression_table, fit_weights,
                                   combine, fit_group_weights, combine_by_group, user_folds, cross_fitted
project/hybrids/weighted_tuning.py Setting, default_grid, HybridData (caches), cv_score, tune, fit_hybrid
project/analysis/coefficients.py   bootstrap_weights, hit_attribution, member_overlap
project/experiments/hybrid.py      CLI: everything above, writes the files of §4
project/tests/test_hybrids.py      hand-computed tests
```
