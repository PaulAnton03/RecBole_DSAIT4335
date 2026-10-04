# DSAIT4335 Recommender Systems — Final Project (group repository)

This repository is our group's fork of the lecturer's course version of RecBole:

- **Lecturer repo (`upstream`)**: <https://github.com/masoudmansoury/RecBole_DSAIT4335> — default branch `main`
- **Our fork (`origin`)**: <https://github.com/PaulAnton03/RecBole_DSAIT4335>

The lecturer's original `README.md` (the RecBole README) is left untouched on purpose, so that pulling lecturer
updates never conflicts with our documentation. **This file is the group guide.** How we work with Git is in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## 1. What is from the lecturer, what is ours

| Path | Owner | Notes |
| --- | --- | --- |
| `recbole/` (incl. `recbole/config/<Model>/ml-100k.yaml`), `dataset/ml-100k/`, `run_recbole.py`, `run_hyper.py`, `run_recbole_group.py`, `save_split.py`, `save_recommendations.py`, `score_from_saved.py`, `significance_test.py`, `generate_split.py`, `requirements.txt`, `setup.py`, `docs/`, `tests/`, `README.md`, … | **lecturer** | Framework supplied by the course. Treat as read-only (see rule below). |
| `project/` | **us** | All code written by our group. |
| `results/`, `figures/`, `report/` | **us** | Experiment outputs and the LaTeX report. |
| `PROJECT.md`, `CONTRIBUTING.md`, `.github/`, `project/requirements.txt`, the *"DSAIT4335 group additions"* block at the end of `.gitignore` | **us** | Setup and documentation. |

> **Rule: do not edit lecturer files.** It keeps `git merge upstream/main` conflict-free. If a lecturer file needs
> different behaviour, wrap or subclass it from `project/` (example: `project/experiments/save_split_seeded.py`),
> or tell the lecturer.

## 2. Repository layout

```
.
├── recbole/ run_*.py save_*.py score_from_saved.py dataset/ …   # lecturer's course RecBole
├── project/                       # OUR code (run from the repo root: python -m project.<...>)
│   ├── configs/                   #   our RecBole/experiment YAML configs (base.yaml = shared protocol, hyper/, models/)
│   ├── models/                    #   Track A: individual models -- train, tune, export (see project/models/README.md)
│   ├── hybrids/                   #   Task 1: hybrid recommenders
│   ├── metrics/                   #   Task 2: accuracy + beyond-accuracy metrics
│   ├── rerankers/                 #   Task 3: diversity / calibration / fairness rerankers
│   ├── analysis/                  #   coefficient, user-group, item-group analyses
│   ├── experiments/               #   runnable entry points: python -m project.experiments.<name>
│   ├── utils/                     #   paths.py, report_assets.py (figure/table writers), data_formats.py (shared file formats)
│   ├── tests/                     #   unit tests: python -m pytest project/tests -q
│   └── requirements.txt           #   extra requirements on top of the lecturer's
├── results/
│   ├── raw/                       # big/reproducible run output   -> git-IGNORED (folder kept)
│   └── processed/                 # small aggregated CSV/JSON     -> tracked
├── figures/generated/             # figures written by Python     -> tracked, consumed by LaTeX
└── report/                        # LaTeX report (Overleaf)
    ├── report.tex                 #   the hand-in report (Times New Roman 12 pt, 1.15 spacing; one page per task)
    ├── figures/                   #   hand-made figures (diagrams, screenshots)
    └── tables/generated/          #   LaTeX tables written by Python -> tracked, consumed by LaTeX
```

## 3. Python environment

The lecturer's instructions (stock RecBole `README.md`): Python ≥ 3.7, install from source with
`pip install -e .`, dependencies in `requirements.txt` / `setup.py`. No Conda environment is prescribed, so we use a
local `.venv/`. Set up and verified with **Python 3.11** (`.venv/` is git-ignored).

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# Only on a machine WITHOUT an NVIDIA GPU: install the small CPU build of torch first (~200 MB
# instead of ~3 GB of CUDA libraries). The lecturer only requires torch>=1.10.0.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e . -r requirements.txt -r project/requirements.txt
```

Check it (should print the version and a path inside this repo):

```bash
python -c "import recbole; print(recbole.__version__, recbole.__file__)"
```

`project/requirements.txt` holds the only additions to the lecturer's requirements, each needed because the
lecturer's files only give lower bounds and pip would otherwise pick versions the code cannot use:

- `ray[tune]` — `recbole.quick_start` does `from ray import tune`, which needs `pyarrow`; plain `ray` lacks it.
- `numpy<2` — `recbole/config/configurator.py` uses `np.float_`, removed in NumPy 2.0 (crashes at start-up).
- `pandas<3` — RecBole fills NaNs with `feat[field].fillna(..., inplace=True)`, a silent no-op under pandas 3.
- `matplotlib` — for `figures/generated/*.pdf`.

Update the environment after pulling changes to any requirements file by re-running the last `pip install` line.

## 4. Running experiments

The lecturer provides one ranking config per model for MovieLens 100K
(`dataset/ml-100k/` is already in the repo — no download):
`BPR EASE FISM ItemKNN LightGCN NeuMF NGCF Pop Random SLIMElastic UserKNN` →
`recbole/config/<Model>/ml-100k.yaml`.

**Basic experiment** (BPR on ml-100k, ~20 s on CPU):

```bash
python run_recbole.py --model=BPR --dataset=ml-100k --config_files=recbole/config/BPR/ml-100k.yaml
```

- **Always pass `--model` together with `--config_files`.** `--model` defaults to `BPR` and takes precedence over the
  `model:` key inside the YAML, so `--config_files=recbole/config/EASE/ml-100k.yaml` alone would silently train BPR.
- Extra `--key=value` arguments override config values, e.g. `--epochs=2 --show_progress=False` for a quick check.
- Checkpoints go to `saved/`, logs to `log/` (both git-ignored). Hyper-parameter search: `run_hyper.py`
  (see the lecturer's README); it can run for a long time, so agree in the group who runs which search, and write
  its output under `results/raw/`, e.g. `--output_file results/raw/hyper_BPR.result`.
- Our own experiments live in `project/experiments/` and are started from the repo root:
  `python -m project.experiments.<name>`. Our configs go in `project/configs/`.

### The shared pipeline (Track A) -- start here

The individual models are trained, tuned and exported by Track A; every other track reads its output files
(frozen split, score matrices, top-50 lists) instead of calling RecBole. Full description, formats and runtimes:
[`project/models/README.md`](project/models/README.md).

```bash
python -m project.experiments.export_split                      # 0. frozen split  (results/raw/splits/)
python -m project.experiments.run_models --mode quick --models all   # 1. untuned scores + top-50 lists
python -m project.experiments.tune_models                       # 2. grid search on validation NDCG@10
python -m project.experiments.run_models --mode tuned           # 3. final (tuned) scores + lists
python -m project.experiments.results_table                     # 4. report table + figure
python -m project.experiments.evaluate_models                   # 5. our own metrics (Track B, see project/metrics/README.md)
```

### Evaluating separately (assignment: "perform the evaluation separately")

The lecturer's scripts dump the split and the top-k lists so metrics can be computed outside RecBole:

```bash
# 1. train (prints the checkpoint path: saved/ml-100k-BPR-<timestamp>.pth)
python run_recbole.py --model=BPR --dataset=ml-100k --config_files=recbole/config/BPR/ml-100k.yaml
# 2. ground-truth split  (use OUR seeded wrapper, see warning below)
python -m project.experiments.save_split_seeded --model BPR --dataset ml-100k \
    --config_files recbole/config/BPR/ml-100k.yaml --output_dir results/raw/splits
# 3. top-k lists for every test user
python save_recommendations.py --model_file saved/<checkpoint>.pth --k 10 --output_dir results/raw/recommendations
# 4. score them
python score_from_saved.py --test_tsv results/raw/splits/ml-100k.test.tsv \
    --rec_json results/raw/recommendations/ml-100k_BPR_top10.json --k 10
```

> **Warning — the lecturer's `save_split.py` is not seeded.** It builds the split without `init_seed`, so its
> "test" split differs from the one the model was trained/reloaded on. Scoring against it gave Recall@10 = 0.017
> where RecBole itself reported 0.2085 for the same checkpoint (only 875 of 9,596 test pairs overlapped). With
> `project.experiments.save_split_seeded` the split is identical (9,596/9,596) and `score_from_saved.py` reproduces
> RecBole's numbers exactly. Also: the comments in the lecturer's configs mention `configs/<Model>/ml-100k.yaml`;
> the real path is `recbole/config/<Model>/ml-100k.yaml`.

## 5. Results, figures and tables

| Location | Content | Git |
| --- | --- | --- |
| `results/raw/` | splits, top-k dumps, per-run outputs — anything large or reproducible | ignored |
| `results/processed/` | small aggregated CSV/JSON the plotting/table scripts read | **tracked** |
| `figures/generated/*.pdf` | figures written by Python | **tracked** (Overleaf needs them) |
| `report/tables/generated/*.tex` | LaTeX tables written by Python | **tracked** (Overleaf needs them) |
| `saved/`, `log/`, `*.pth`, `*.log`, … | checkpoints, logs | ignored |

Write report assets through the helpers so names stay stable and files are byte-reproducible (no timestamps, so
re-running an experiment gives a clean diff instead of noise):

```python
from project.utils.report_assets import save_figure, save_latex_table

save_figure(fig, "model_comparison")     # -> figures/generated/model_comparison.pdf
save_latex_table(df, "model_results")    # -> report/tables/generated/model_results.tex  (booktabs tabular, escaped)
```

Agreed file names (the report already references them): `model_comparison`, `hybrid_coefficients`,
`user_group_analysis` (figures) and `model_results`, `tuning_summary`, `reranker_results`, `beyond_accuracy_results`, `metric_validation` (tables). Add new ones with a fixed,
descriptive `snake_case` name — never put a date, run id or seed in the name; overwrite in place. Commit a generated
file together with the code change that produced it. Generated tables contain only the `tabular`; the caption and
label live in `report/report.tex`.

## 6. The report and Overleaf

`report/report.tex` is the hand-in report (title page, Tasks 1–3, appendix in one file; formatted as the brief
requires: Times New Roman, 12 pt, line spacing 1.15, at most one page and 200 words of discussion per task).
Each member writes the subsections of the sub-tasks they own; the owner is named in a comment above each one.
Generated assets are included with two macros defined in `report.tex`:

```latex
\generatedfigure[width=0.9\linewidth]{model_comparison}   % ../figures/generated/model_comparison.pdf
\generatedtable{model_results}                            % tables/generated/model_results.tex
```

If a generated file has not been committed yet, a red *TODO* box is shown instead, so the report always compiles.
Fill in every `\TODO{...}`; the skeleton contains **no** results.

Local build: `cd report && tectonic report.tex` (or `latexmk -pdf -outdir=build report.tex` with a TeX distribution)
(`report/build/` and `report/*.pdf` are git-ignored).

**Connecting Overleaf (once, by one member; needs an Overleaf plan with GitHub sync):** Overleaf → *New Project* →
*Import from GitHub* → select `PaulAnton03/RecBole_DSAIT4335`; then *Menu → Main document → `report/report.tex`*.
Overleaf imports the whole repository (~25 MB), so the RecBole code also shows up in its file tree; ignore it.

**How data flows** (Overleaf's GitHub sync is *manual* in both directions):

```
run experiment → figures/generated/*.pdf, report/tables/generated/*.tex
   → commit + push feature branch → pull request → merge into main
   → Overleaf: Menu → Sync → GitHub → "Pull GitHub changes into Overleaf" → recompile
```

Text edited in Overleaf is sent back with *"Push Overleaf changes to GitHub"*, which commits to the synced branch
(`main`). To avoid conflicts, pull `main` before you start report work, and don't edit the same `.tex` file in
Overleaf and on a branch at the same time.

## 7. Getting updates from the lecturer

The lecturer repo's default branch is **`main`**. Keep our `main` up to date (no local edits to lecturer files → no
conflicts expected):

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

Then bring feature branches up to date with `git merge main`, and re-run the `pip install` line from §3 if
`requirements.txt` changed. **Never push to `upstream`** (that is the lecturer's repo).

## 8. Known quirks (all from the lecturer's code, harmless unless noted)

- `WARNING Could not save embeddings: BPR.forward() missing 2 required positional arguments` after each epoch.
- `FutureWarning` about `torch.cuda.amp.GradScaler` (`trainer.py`).
- `FutureWarning` about chained assignment in `dataset.py` (lines 648/650) — this is the `fillna(inplace=True)` that
  pandas 3 would silently skip, which is why `pandas<3` is pinned (§3).
- `use_gpu: True` is the default, but RecBole falls back to CPU automatically when CUDA is not available.
