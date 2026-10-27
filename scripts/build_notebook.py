"""Build and execute notebooks/exploratory_analysis.ipynb from the project modules.

    pip install nbformat nbclient ipykernel
    python scripts/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
cells = [
    ("md", "# Exploratory analysis — F1 Race Intelligence\n"
           "Executed against the processed dataset produced by `python -m src.data_pipeline`. "
           "All numbers below are computed, not typed in. Correlations are descriptive, not causal."),
    ("code", "import sys, json\nsys.path.insert(0, '..')\nimport pandas as pd\n"
             "from src import eda\nfrom src.config import resolve\n"
             "df = pd.read_csv(resolve('data/processed/model_dataset.csv'))\n"
             "study = pd.read_csv(resolve('data/processed/features_study.csv'))\n"
             "print(df.shape, study.shape)"),
    ("md", "## Data quality"),
    ("code", "q = eda.data_quality(df)\nq[q['missing'] > 0]"),
    ("code", "print('duplicate driver-race keys:', df.duplicated(['season','round','driver_id']).sum())\n"
             "json.load(open(resolve('data/processed/cleaning_summary.json')))['exclusions_by_reason']"),
    ("md", "## Class balance by chronological split"),
    ("code", "eda.class_balance(df)"),
    ("md", "## Grid slot and podium rate"),
    ("code", "eda.podium_rate_by_grid(df).round(3)"),
    ("md", "## Experience and missing history"),
    ("code", "eda.experience_vs_podium(df)"),
    ("code", "eda.missing_history(study)"),
    ("md", "## Circuits where podium finishers start furthest back"),
    ("code", "eda.circuit_patterns(df).head(10)"),
    ("md", "## Spearman correlation with the target"),
    ("code", "eda.correlation(df)['podium'].sort_values()"),
    ("md", "## Podium rate by constructor recent points-per-race decile"),
    ("code", "eda.feature_target_deciles(df, 'con_ppr_5').round(3)"),
]
nb = nbf.v4.new_notebook()
nb["cells"] = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
out = ROOT / "notebooks" / "exploratory_analysis.ipynb"
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).execute()
nbf.write(nb, out)
print("wrote", out)
