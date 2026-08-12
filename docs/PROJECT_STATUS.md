# Repository status and reproducibility roadmap

## Current state

This repository preserves the original course report and has been reorganized to document the problem, data source, reported metrics, and intended end-to-end workflow.

The original experiment notebooks, source code, trained models, and derived datasets are not currently available in the repository. Therefore, the figures and metrics in the report should be treated as **reported experimental results**, not yet as one-command reproducible outputs.

## What to add next

### 1. Restore the original work

Collect any existing files from team members before rebuilding:

- Jupyter notebooks (`.ipynb`)
- Python scripts (`.py`)
- Feature lists and preprocessing code
- Saved models (`.pkl`, `.joblib`)
- SHAP charts and confusion matrices
- A `requirements.txt` or environment file used at the time

Commit notebooks after clearing outputs that include personal paths or credentials. Never upload Kaggle API tokens, raw customer-like data, or large trained-model files without deciding on a release method.

### 2. Make the experiment reproducible

Create the notebooks below, each with a short narrative and fixed random seed:

| Notebook | Minimum contents |
|---|---|
| `01_eda.ipynb` | target distribution, missing values, feature distributions, data-quality decisions |
| `02_feature_engineering.ipynb` | categorical encoding, missing-value handling, train/test alignment, engineered features |
| `03_model_comparison.ipynb` | train all candidate models, Grid Search, metrics table, ROC curves, confusion matrices |
| `04_shap_explainability.ipynb` | global feature importance, beeswarm/dependence plots, 2-3 local explanations |

### 3. Improve the business layer

Add a final `05_threshold_and_business_impact.ipynb` that:

- compares thresholds rather than using only the default 0.50 cutoff;
- defines the cost of a false negative (missed default) and false positive (rejected good customer);
- recommends a threshold based on that cost trade-off;
- reports approval rate, recall, precision, and expected cost at the selected threshold.

### 4. Publish portfolio evidence

Export the following into `reports/figures/` and embed them in the README:

- model comparison chart;
- ROC curve for the top models;
- XGBoost confusion matrix at the chosen threshold;
- SHAP global importance chart;
- one anonymized local explanation.

## Definition of done

The repository is portfolio-ready when a reviewer can clone it, install dependencies, obtain the public Kaggle dataset, run the notebooks in order, and see the evaluation and SHAP figures reproduced with documented differences if package versions change.
