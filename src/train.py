"""Train reproducible baseline, Random Forest, and XGBoost credit-risk models.

Example:
    python -m src.train --data data/raw/application_train.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import build_preprocessor, load_application_data
from src.metrics import choose_f1_threshold, classification_metrics


def build_models(random_state: int) -> dict[str, object]:
    """Return models used in the initial reproducible comparison."""
    from xgboost import XGBClassifier

    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1_000, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=6.75,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit-default risk models.")
    parser.add_argument("--data", type=Path, required=True, help="Path to application_train.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest", "xgboost"])
    parser.add_argument("--sample-size", type=int, default=None, help="Optional stratified development sample.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    features, target = load_application_data(args.data)
    if args.sample_size and args.sample_size < len(features):
        features, _, target, _ = train_test_split(
            features, target, train_size=args.sample_size, stratify=target, random_state=args.random_state
        )

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features, target, test_size=0.30, stratify=target, random_state=args.random_state
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.50, stratify=y_holdout, random_state=args.random_state
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | str]] = []
    models = build_models(args.random_state)
    unknown = set(args.models) - set(models)
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown))}")

    for model_name in args.models:
        pipeline = Pipeline([( "preprocessor", build_preprocessor(x_train)), ("model", models[model_name])])
        pipeline.fit(x_train, y_train)

        validation_probability = pipeline.predict_proba(x_validation)[:, 1]
        threshold = choose_f1_threshold(y_validation.to_numpy(), validation_probability)
        test_probability = pipeline.predict_proba(x_test)[:, 1]
        metrics = classification_metrics(y_test.to_numpy(), test_probability, threshold)
        metrics["model"] = model_name
        results.append(metrics)

        joblib.dump(pipeline, args.output_dir / f"{model_name}.joblib")
        print(f"{model_name}: {metrics}")

    result_frame = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    result_frame.to_csv(args.output_dir / "model_metrics.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )
    print(f"Saved metrics and models to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
