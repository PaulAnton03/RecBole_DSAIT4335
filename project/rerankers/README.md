# Track E — re-rankers (Tasks 3.1, 3.2)

Owner: Person E ("Re-rankers"). Post-processing: every re-ranker re-orders a list's top-50 candidates per user,
trading the model's relevance against one beyond-accuracy objective. No model is retrained.

## 1. Usage

```bash
python -m project.experiments.rerank_models                                   # SLIMElastic + LightGCN, all objectives
python -m project.experiments.rerank_models --models SLIMElastic,LightGCN,WeightedHybrid
python -m pytest project/tests/test_rerankers.py -q
```

From Python (any list in the shared format: model, hybrid, or another re-ranked list):

```python
from project.metrics import EvaluationContext
from project.rerankers import MMR, Calibrated, PopCalibrated, XQuAD, rerank
from project.utils.data_formats import load_recommendations

ctx = EvaluationContext.for_split("valid")                 # the split of the list
recs = rerank(load_recommendations("EASE", "valid"), Calibrated(), lam=0.5, ctx=ctx)   # 50 rows per user
```

For Task 3.3 (`project/experiments/order_comparison.py`), `list_reranker` wraps any objective as the plain
`fn(recs, k_out, k_in)` function the pipeline-order comparison expects (re-rank the top `k_in`, keep the top `k_out`):

```bash
python -m project.experiments.order_comparison --objective xQuAD --lambda 0.5 --split valid
```

```python
from project.rerankers import list_reranker, make_objective
reranker_fn = list_reranker(make_objective("Calibrated"), lam=0.4, ctx=ctx)   # ctx of the lists' split
```

## 2. The loop (`greedy.py`)

With `L` the items placed so far, the next slot gets `argmax_{i ∉ L} (1 − λ)·rel(i) + λ·gain(i | L)`
(Diversity slides 10–11, Calibration slides 13–18, Fairness slide 48).

| Choice | What we do | Why |
| --- | --- | --- |
| `rel(i)` | model score, min-max normalised over the user's candidates | one λ grid means the same for every model and hybrid |
| λ | weights the beyond-accuracy term; 0 = model order, 1 = objective only | as on the diversity / calibration slides (the fairness slide puts λ on relevance) |
| ties | better original rank wins | λ = 0 reproduces the input exactly |
| output | all 50 candidates re-ordered, `score = (n − rank + 1) / n` | shared format; a score that follows the new order (Task 3.3 combines re-ranked lists) |
| `depth` | greedy for the first `depth` slots only, rest in original order | the λ sweep needs the top 10 only (identical top 10, ~2× faster) |

## 3. Objectives (`objectives.py`)

Each one reuses Track B's metric code, so it optimises exactly what is reported. Tests check `gain = −metric`.

| Name | Goal | `gain(i | L)` | Target metric |
| --- | --- | --- | --- |
| `MMR` | diversity | mean genre-Jaccard distance of `i` to `L` (0 for the first slot) | ILD@10 ↑ |
| `Calibrated` | calibration (Steck 2018) | `−KL(p(g|u) ‖ q̃(g|L ∪ i))`, α = 0.01 | MC_KL@10 ↓ |
| `PopCalibrated` | user-side fairness (Abdollahpouri et al. 2021) | `−JSD` of head/mid/tail/unseen mix: profile vs `L ∪ i` | UPD@10 ↓ |
| `xQuAD` | item-side fairness (Abdollahpouri et al. 2019) | smooth xQuAD, short head (head+mid) vs long tail (tail+unseen), P(c|u) = ½ | TailShare@10 ↑ |

`XQuAD(personalised=True)` weights the two categories by the user's own long-tail share (the paper's version; then it
is close to `PopCalibrated`).

## 4. Choosing λ (`tradeoff.py`)

λ is tuned on **validation** (grid 0, 0.1, …, 1) and reported on **test**. `knee` scales both axes to [0, 1]
(target-metric improvement, NDCG@10) and takes the point furthest above the line from λ = 0 to λ = 1; if nothing
improves, λ = 0. The knee is scale-free: it can accept a large NDCG loss if the target keeps improving steeply —
always check it against `figures/generated/reranker_tradeoff.pdf`.

## 5. Outputs

| File | Content |
| --- | --- |
| `results/processed/reranker_sweep_valid.csv` | every metric per model × objective × λ (validation) |
| `results/processed/reranker_results.csv` | test metrics of each base list and its re-ranked lists at the chosen λ |
| `results/processed/reranker_pool_test.csv` | metrics of the full top 50 (k = 50): how much diversity / long tail the candidates offer |
| `results/raw/recommendations/<Model>+<Objective>_{valid,test}_top50.csv` | re-ranked lists (Track B: Task 3.4; Track D: Task 3.3) |
| `report/tables/generated/reranker_results.tex`, `figures/generated/reranker_tradeoff.pdf` | report assets |
