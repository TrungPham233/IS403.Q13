# Report artifacts

This folder contains portfolio figures exported from reproducible runs of the code in `src/`.

Run the training pipeline first, then generate the comparison chart:

```bash
python -m src.report --metrics artifacts/model_metrics.csv
```

Generated SVG charts are versioned only when they come from a documented training run. Raw data, serialized models, and temporary artifacts remain excluded from Git.

## Included reproducibility run

`portfolio_sample_50000_metrics.csv` and `portfolio_sample_50000_run_config.json` document a stratified 50,000-row run executed on the public `application_train.csv` dataset with random state 42. Its [model-comparison chart](figures/model_comparison.svg) is a development-scale demonstration of the repository workflow, not a substitute for the full-study metrics reported in the original PDF.
