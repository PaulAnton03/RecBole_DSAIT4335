# Contributing (group workflow)

```
upstream (lecturer)  →  our fork (origin)  →  feature branches  →  pull requests  →  main  →  Overleaf
```

Setup, running experiments and the report are described in [`PROJECT.md`](PROJECT.md).

## Remotes

| Remote | URL | Use |
| --- | --- | --- |
| `origin` | `https://github.com/PaulAnton03/RecBole_DSAIT4335.git` | our fork — push here |
| `upstream` | `https://github.com/masoudmansoury/RecBole_DSAIT4335.git` | lecturer — **fetch only** |

First time on a new machine:

```bash
git clone https://github.com/PaulAnton03/RecBole_DSAIT4335.git
cd RecBole_DSAIT4335
git remote add upstream https://github.com/masoudmansoury/RecBole_DSAIT4335.git
gh repo set-default PaulAnton03/RecBole_DSAIT4335     # so `gh pr create` targets OUR fork
```

Collaborators need write access to the fork: *GitHub → Settings → Collaborators* (owner does this once).

> **Pull requests must target our fork.** For a fork, GitHub's *"Compare & pull request"* button and `gh pr create`
> default to the *lecturer's* repository. Check that the base repository reads `PaulAnton03/RecBole_DSAIT4335`,
> base branch `main`. Never open PRs against `masoudmansoury/RecBole_DSAIT4335`.

## Branches

Never commit directly to `main`. One short-lived branch per piece of work, named `<type>/<topic>`:

| Prefix | Use | Examples |
| --- | --- | --- |
| `feature/` | new code in `project/` | `feature/metrics`, `feature/hybrid`, `feature/reranking` |
| `experiment/` | running experiments, producing results/figures/tables | `experiment/model-comparison` |
| `report/` | LaTeX changes | `report/task2-analysis` |
| `fix/` | bug fixes | `fix/ndcg-tie-breaking` |
| `chore/` | tooling, docs, config | `chore/update-requirements` |

```bash
git checkout main && git pull origin main
git checkout -b feature/metrics
# ... work, commit small and often ...
git push -u origin feature/metrics
gh pr create --base main            # or use the GitHub UI (check the base repository, see above)
```

## Pull requests

- Small and focused; one reviewer from the group approves before merging (squash or merge commit — either is fine).
- Update from `main` before requesting review: `git fetch origin && git merge origin/main`.
- If an experiment changed `figures/generated/` or `report/tables/generated/`, commit the regenerated files in the
  same PR as the code that produced them (Overleaf only sees what is on `main`).
- Never commit datasets, checkpoints (`saved/`, `*.pth`), logs, `.venv/`, `results/raw/` or credentials.
  Check `git status` and `git diff --stat` before committing.
- Do not edit lecturer files (see `PROJECT.md` §1); put wrappers and extensions in `project/`.
- Commit messages: short imperative summary, optional prefix (`feat:`, `fix:`, `docs:`, `chore:`, `exp:`, `report:`).

## Staying in sync with the lecturer

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

Then merge `main` into your open feature branches. Never push to `upstream`, never force-push `main`.

## Report / Overleaf

Only `main` reaches Overleaf, and Overleaf's GitHub sync is manual (*Menu → Sync → GitHub*). After merging a PR that
changes the report or generated assets, pull the changes into Overleaf. Text edited in Overleaf is pushed to `main`
with *Push Overleaf changes to GitHub*: run `git pull origin main` before starting new report work.
