# `project/` — our group's code

Everything outside this folder (`recbole/`, `run_*.py`, `save_*.py`, `dataset/`, …) is the lecturer's framework;
**write our code here** and import RecBole from it. Run from the repository root:

```bash
python -m project.experiments.<name>
```

| Folder | Content |
| --- | --- |
| `configs/` | our RecBole / experiment YAML configs |
| `hybrids/` | hybrid recommenders (Task 1) |
| `metrics/` | accuracy and beyond-accuracy metrics (Task 2) |
| `rerankers/` | diversity, calibration and fairness rerankers (Task 3) |
| `analysis/` | coefficient, user-group and item-group analyses |
| `experiments/` | runnable entry points (e.g. `save_split_seeded.py`) |
| `utils/` | `paths.py` (repo-relative locations), `report_assets.py` (`save_figure`, `save_latex_table`) |

Report assets: `save_figure(fig, "model_comparison")` → `figures/generated/model_comparison.pdf`,
`save_latex_table(df, "model_results")` → `report/tables/generated/model_results.tex`. See [`../PROJECT.md`](../PROJECT.md).
