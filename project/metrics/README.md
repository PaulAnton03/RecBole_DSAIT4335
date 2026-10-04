# Track B — evaluation metrics (Task 2.1) and group analysis (Task 2.5)

Owner: Person B ("Evaluation"). Our own implementation of every accuracy and beyond-accuracy metric, as the brief
requires ("implement all metrics yourself separately for consistency of your evaluation"). RecBole only produces
the recommendation lists (Track A); **no reported number comes from RecBole's evaluator**. RecBole is used only
to *check* the implementation where it has the same metric (§5).

## 1. Usage

```bash
python -m project.experiments.evaluate_models                  # every exported list, valid + test (≈ 30 s)
python -m project.experiments.evaluate_models --split test --names EASE,MyHybrid
python -m project.experiments.evaluate_models --min-rating 4   # relevance = held-out rating >= 4 (appendix)
python -m project.experiments.group_analysis                   # Task 2.5: user and item groups (≈ 30 s, see §7)
python -m pytest project/tests -q                              # hand-computed tests
```

| Output | Content |
| --- | --- |
| `results/processed/metrics_{valid,test}.csv` | one row per list: every metric below + user counts (tracked) |
| `results/raw/metrics/<Name>_<split>_per_user.csv` | per-user values + activity and taste group |
| `results/processed/metric_validation.csv` | our value vs. RecBole's for every list and checked metric |
| `report/tables/generated/beyond_accuracy_results.tex` | beyond-accuracy metrics on test (report, Task 2.1) |
| `report/tables/generated/metric_validation.tex` | largest difference to RecBole per metric (appendix) |

From Python, any list in the shared format (`user_id, item_id, rank[, score]`) — model, hybrid or re-ranker:

```python
from project.metrics import EvaluationContext, evaluate
ctx = EvaluationContext.for_split("valid")       # cached; tune on valid, report on test
res = evaluate(my_recs_df, ctx, k=10, name="MyHybrid")
res.summary["ndcg@10"], res.summary["ild@10"]    # pd.Series with every metric
res.per_user                                     # one row per user (index user_id)
```

One call takes ≈ 0.2 s, so it can sit inside a tuning or λ-sweep loop.

## 2. Evaluation contract

| Symbol | Definition (all from the frozen split, `project/utils/data_formats.py`) |
| --- | --- |
| `L_u@K` | the user's list, `rank <= K` (K = 10), ordered by rank |
| `T_u` | relevant items: all held-out interactions of the split (agreed protocol); with `--min-rating r`, those rated ≥ r |
| masked | items that may not be recommended: train for `valid`, train + valid for `test` — exactly what RecBole masks |
| `H_u` | the user's training interactions (profile for calibration, UPD and the user groups) |
| `I` | catalogue: all 1,682 ML-100K movies (`dataset/ml-100k/ml-100k.item`); no padding/reserved ids |
| `U` | evaluated users: every user with `T_u ≠ ∅` (943) |
| `c_i` | popularity: number of training interactions of `i` (all users) |

Everything that describes users or items (popularity, profiles, all groups) comes from the **training split
only**: the data every model was trained on, identical for `valid` and `test`, frozen across models. Held-out
interactions are never used.

- **Aggregation.** Per-user metrics are computed first, then macro-averaged over `U`. Coverage, Gini, entropy and
  group exposure are computed once over the lists of all of `U` (no batches).
- **Strict input.** A list with a duplicate item, a duplicate rank, an unknown item, or a masked item raises an
  error (the latter usually means a `valid` list is scored against `test`).
- **Edge cases.** A user of `U` without a list gets an empty list: accuracy 0, list-based beyond-accuracy metrics
  undefined. Undefined values (ILD of < 2 items, any list metric of an empty list, miscalibration with an empty
  history) are NaN, left out of the mean and reported via `users_missing_list` / `users_short_list`. List users
  outside `U` are ignored (`users_ignored`). Precision always divides by K, so short lists are not rewarded.
  On the frozen split none of these cases occurs (every user has 50 candidates and ≥ 1 held-out item).

## 3. Metrics

Lecture = DSAIT4335 slides. "Choice" marks a convention the slides leave open.

| Key | Metric | Definition | Better | Source / choice |
| --- | --- | --- | --- | --- |
| `precision@K` | Precision | `|L ∩ T| / K` | ↑ | Evaluation 1 |
| `recall@K` | Recall | `|L ∩ T| / |T|` | ↑ | Evaluation 1 |
| `f1@K` | F1 | harmonic mean of the user's P and R (0 if both 0) | ↑ | Evaluation 1 |
| `hit@K` | Hit rate | 1 if `|L ∩ T| > 0` | ↑ | |
| `ndcg@K` | NDCG | `Σ_r rel_r / log2(r+1)` ÷ `Σ_{r ≤ min(K,|T|)} 1 / log2(r+1)`, binary rel | ↑ | Evaluation 2, sl. 7 |
| `mrr@K` | MRR | `1 / rank` of the first relevant item, 0 if none | ↑ | Evaluation 2, sl. 11 |
| `map@K` | MAP | `Σ_r P@r · rel_r / min(K, |T|)` | ↑ | Evaluation 2, sl. 13; **choice:** denominator (below) |
| `ild@K` | Intra-list diversity | mean Jaccard distance of the genre sets over all item pairs of the list | ↑ | Diversity, sl. 9; **choice:** genre Jaccard |
| `coverage@K` | Catalogue coverage | `|∪_u L_u| / |I|` (fraction) | ↑ | Diversity, sl. 12; Fairness, sl. 42 |
| `novelty@K` | Novelty | mean over the list of `−log2(c_i / Σ_j c_j)` (bits) | ↑ | Diversity, sl. 15; **choice:** `c_i = 0` → 1 |
| `serendipity@K` | Serendipity | `|{i ∈ L : i ∈ T, i ∉ Pop_u@K}| / K` | ↑ | Diversity, sl. 16–17; **choice:** formula below |
| `avgpop@K` | Average popularity | mean over the list of `c_i / |users|` | ↓ | popularity bias; RecBole's AveragePopularity / |users| |
| `tailshare@K` | Long-tail share | share of the list in the tail or unseen group | ↑ | **choice:** groups below |
| `miscalibration@K` | Genre miscalibration | `KL(p(g|u) ‖ (1−α) q(g|u) + α p(g|u))`, α = 0.01, natural log | ↓ | Calibration, sl. 9–12 (Steck 2018) |
| `upd@K` | User popularity deviation | `JSD(P(H_u), P(L_u))`, P = distribution over head/mid/tail/unseen, log2 | ↓ | Fairness, sl. 39; **choice:** P and JSD |
| `gru_ndcg@K` | Group recommendation unfairness | max − min of the mean NDCG over the low/medium/high activity groups (+ every mean and size) | ↓ | Fairness, sl. 38; **choice:** groups, max−min |
| `gini@K` | Gini index of exposure | `Σ_j (2j − n − 1) e_(j) / (n Σ e)`, `e_i` = lists containing `i`, all `n = |I|` items | ↓ | Fairness, sl. 42; standard formula |
| `entropy@K` | Normalised exposure entropy | `−Σ s_i ln s_i / ln |I|`, `s_i = e_i / Σ e` | ↑ | Fairness, sl. 42 |
| `exposure_{head,mid,tail,unseen}@K` | Group exposure | mean over the group's items of `Σ_u 1/log2(1 + pos_u(i))`, divided by `|U|` | — | Fairness, sl. 43 (demographic parity) |

**Conventions in detail**

- **MAP denominator.** Slide 13 says "number of relevant items", but the worked example on slide 14 divides by
  the number of hits in the list (3 of 5). We divide by `min(K, |T_u|)` (RecBole's choice): a user with more
  relevant items than slots is not penalised for what cannot fit. The test `test_lecture_mrr_and_map_convention`
  pins this down.
- **ILD distance.** The slide allows genre overlap or embedding distance. We use the Jaccard distance between
  genre sets for every model, because model-specific embeddings would change the yardstick between models.
  ML-100K movies have 1.7 genres on average (19 genres; the 2 movies labelled `unknown` keep it as a genre), so
  most pairs share no genre and ILD varies within a narrow band. Report differences, not absolute levels.
- **Novelty.** `pop(i)` is the share of all training interactions on `i` (slide 15), *not* the share of users. An
  item never seen in training would have infinite surprisal; it gets the count 1, i.e. the novelty of the
  rarest observed item.
- **Serendipity.** "Unexpected and useful" (slides 16–17), with unexpectedness measured against a popularity
  baseline as the slide suggests: an item counts when it is relevant *and* not among the `K` most popular items
  the user may be recommended (`Pop_u@K`: by training counts, masked items skipped, ties by catalogue order). An exact most-popular list scores 0;
  RecBole's `Pop` scores about 0.02, because it ranks by the number of training batches that contain a film
  (including sampled negatives), not by its number of interactions.
- **Popularity groups.** Items sorted by `c_i`: **head** = the most popular items that together hold ≥ 20 % of
  the training interactions, **tail** = the least popular items that together hold ≤ 20 %, **mid** = the rest;
  **unseen** = no training interaction (kept apart: never observed is not the same as rare). Items with equal
  counts always share a group. Head 58 items (`c_i ≥ 216`), mid 479, tail 1,118 (`1 ≤ c_i ≤ 48`), unseen 27.
  The same groups are the item groups of Task 2.5 (§7).
- **Miscalibration.** `p(g|i)` puts `1/m` on each of an item's `m` genres (KL needs distributions, not multi-hot
  vectors). History and list weights are uniform; `calibration.list_profile(..., rank_weighted=True)` gives the
  rank-discounted variant the slide mentions. Direction `KL(p ‖ q̃)` as on slide 12.
- **UPD.** The slide leaves `P` and `dist` open. `P` = share of the profile's / list's items in head, mid,
  tail and unseen; `dist` = Jensen–Shannon divergence (base 2, in [0, 1], finite when a group is missing on one side).
- **GRU groups.** The activity tertiles of §7 (low / medium / high training interactions). With three groups we
  report the max − min extension (the slide's formula for two groups), always with every group's mean and size.
  Caveat: with a random per-user split, active users also have more held-out items, which by itself raises
  their NDCG (Random: 0.016 vs 0.004); §7 compares models within groups instead.
- **Exposure.** Counted over the whole catalogue, so never-recommended items count with `e_i = 0`. Gini's
  maximum is `(n − 1) / n`, not 1. RecBole's own run divides coverage and Gini by `|I| + 1` (it counts the
  `[PAD]` id); we use `|I|`.

## 4. Building blocks for the other tracks

The re-rankers and hybrids should optimise exactly what is measured here.

| Need | Function |
| --- | --- |
| movie genres / catalogue | `lookups.load_item_genres()`, `lookups.load_catalogue()`, `ctx.genres`, `ctx.genre_names` |
| item popularity, head/mid/tail/unseen | `ctx.counts`, `ctx.item_groups` (`lookups.HEAD/MID/TAIL/UNSEEN`), `lookups.popularity_groups(counts)` |
| user groups (activity, taste) | `ctx.user_groups` (DataFrame), `lookups.user_groups(train, item_groups, catalogue)` |
| group analysis of any lists | `project.analysis.group_analysis.analyse({name: recs}, ctx)` (§7) |
| MMR distance | `diversity.jaccard_distance_matrix(ctx.genres)` (I × I) |
| calibration objective | `calibration.item_genre_distribution`, `calibration.profile`, `calibration.kl_miscalibration` |
| exposure of a set of lists | `fairness.item_exposure(idx, n_items, position_discount=...)` |

`ctx = EvaluationContext.for_split(split)`; arrays are aligned to `ctx.catalogue` (items) and `ctx.users` (users),
and `ctx.item_pos[item_id]` gives an item's index.

## 5. Validation against RecBole

`evaluate_models` compares, on every exported list and both splits:

- NDCG, Recall, Precision, MRR, Hit and MAP@10 with the values RecBole reported for the same checkpoint
  (`results/processed/recbole_metrics_{tuned,quick}.csv`; Track A guarantees ranks 1–10 are RecBole's lists);
- coverage, Gini, entropy and average popularity with RecBole's metric classes applied to our lists and our
  catalogue size.

All differences are at RecBole's rounding (4 decimals) or floating-point level
(`results/processed/metric_validation.csv`). The metrics RecBole lacks (ILD, novelty, serendipity, tail share,
miscalibration, UPD, GRU, group exposure, item-group metrics) are pinned down by hand-computed cases in
`project/tests/test_metrics.py` and `project/tests/test_group_analysis.py`, including the lecture's worked examples.

## 6. Files

```
project/metrics/evaluate.py      EvaluationContext, evaluate(), evaluate_saved()
project/metrics/accuracy.py      Precision, Recall, F1, Hit, NDCG, MRR, MAP
project/metrics/diversity.py     ILD, coverage, Jaccard distances
project/metrics/novelty.py       novelty, serendipity, popularity-baseline lists
project/metrics/calibration.py   genre distributions, KL miscalibration
project/metrics/popularity.py    average popularity, tail share, UPD
project/metrics/fairness.py      GRU, exposure, Gini, entropy, group exposure
project/metrics/item_groups.py   item-group recall and exposure (Task 2.5)
project/metrics/lookups.py       genres, popularity, item groups, user activity and taste groups
project/analysis/group_analysis.py       group means + bootstrap intervals, gaps, paired comparisons, alignment
project/experiments/evaluate_models.py   CLI: evaluate all lists, validate, write report tables
project/experiments/group_analysis.py    CLI: Task 2.5 tables, figures and CSVs
project/tests/test_metrics.py, test_group_analysis.py   hand-computed tests
```

## 7. User and item groups (Task 2.5)

`python -m project.experiments.group_analysis` analyses every exported list (models, and hybrids or re-ranked
lists as soon as they are exported; `--names` restricts, `--figure-names` picks the lists in the figures).

**Groups** (frozen, training split only, `results/processed/group_definitions.json`):

| Dimension | Groups | Rule |
| --- | --- | --- |
| user activity | low 321 / medium 308 / high 314 | tertiles of training interactions: ≤ 34, 35–92, ≥ 93 |
| user taste | niche 314 / mixed 318 / mainstream 311 | tertiles of the share of head items in the training interactions: ≤ 19 %, 20–29 %, ≥ 30 % |
| items | head 58 / mid 479 / tail 1,118 / unseen 27 | cumulative training-interaction mass 20 / 60 / 20 %; unseen = no training interaction |

Tertile cuts are value thresholds, so equal values share a group (sizes are approximately equal). Activity and
taste are kept separate (an active user can have niche taste; Pearson r = −0.53, 62 % of niche users are highly
active); their cross-table is `user_group_intersection_<split>.csv`. These thresholds are experimental choices,
not lecture requirements; they were fixed before looking at any group result.

**Outputs** (`results/processed/`, both splits — use `valid` for design decisions such as Task 2.6, `test` for
reporting):

| File | Content |
| --- | --- |
| `user_group_metrics_<split>.csv` | per list × dimension × group × metric (NDCG, Recall, ILD, novelty, MC, UPD, TailShare): mean, 95 % bootstrap CI, users |
| `user_group_gaps_<split>.csv` | max − min gap across a dimension's groups per metric (GRU for NDCG), with the groups involved |
| `user_group_pairs_<split>.csv` | every two lists, per group and overall: paired mean NDCG / Recall difference A − B with 95 % bootstrap CI |
| `item_group_metrics_<split>.csv` | per list × item group: micro recall (+CI, support = held-out pairs), hit share, slot share (plain and rank-discounted), within-group coverage, lists per item |
| `popularity_alignment_<split>.csv` | per list × taste group: mean head/mid/tail/unseen share of the profiles and of the lists |
| `user_group_intersection_<split>.csv` | activity × taste cells: users, NDCG, Recall |

Report assets (test): `figures/generated/user_group_analysis.pdf`, `figures/generated/popularity_alignment.pdf`,
`report/tables/generated/{group_definitions,user_group_results,item_group_results}.tex`.

**Reading the results.** With a random per-user hold-out, users with longer histories also have more held-out
items (high activity 22 vs low 2.5; niche 18 vs mainstream 4). NDCG@10 rises and Recall@10 falls with that number
for reasons unrelated to the model, so raw group gaps are not evidence of unfair treatment; compare models within
a group (`user_group_pairs`) or against the baselines in the same group. Item-group recall is micro-averaged and
keeps the original ranks (no list is filtered to one group). Small groups (unseen: 21 held-out test pairs) are
unstable; always read the support. Trust a difference between models only if it holds on both splits: on test,
LightGCN beat UserKNN by 0.040 NDCG for high-activity users, but on validation the two are tied in every group.
