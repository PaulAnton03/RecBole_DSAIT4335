"""Repository locations, independent of the current working directory."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = REPO_ROOT / "dataset"  # lecturer-supplied (ml-100k)
PROJECT_CONFIGS_DIR = REPO_ROOT / "project" / "configs"

RESULTS_RAW_DIR = REPO_ROOT / "results" / "raw"  # git-ignored, large/reproducible
RESULTS_PROCESSED_DIR = REPO_ROOT / "results" / "processed"  # tracked, small aggregates

# Consumed by the LaTeX report through Git/Overleaf. Both are tracked on purpose.
FIGURES_GENERATED_DIR = REPO_ROOT / "figures" / "generated"
TABLES_GENERATED_DIR = REPO_ROOT / "report" / "tables" / "generated"
