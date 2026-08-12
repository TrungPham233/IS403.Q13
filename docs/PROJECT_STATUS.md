# Repository status and reproducibility roadmap

## Current state

This repository preserves the original course report and includes a clean local training pipeline for the Home Credit `application_train.csv` dataset. It is organized to document the problem, data source, reported metrics, and end-to-end workflow.

The original experiment notebooks used Google Colab and a Google Drive preprocessed-data file, so they are not copied directly into the repository. The local code in `src/` removes those personal-path dependencies. The figures and metrics in the report should still be treated as **reported experimental results** until this cleaned pipeline has been run with a pinned environment and its output figures are committed.

## What to add next

### 1. Restore the original work

Keep the following original files safely outside the Git repository as references:

- Jupyter notebooks (`.ipynb`)
- Python scripts (`.py`)
- Feature lists and preprocessing code
- Saved models (`.pkl`, `.joblib`)
- SHAP charts and confusion matrices
- A `requirements.txt` or environment file used at the time

If any legacy notebook is added later, clear outputs that include personal paths or credentials. Never upload Kaggle API tokens, raw data, or large trained-model files.

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
