"""Model training pipeline for cloud cost anomaly detection."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import RANDOM_STATE, setup_directories, set_random_seed, print_separator
from src.data_loading import load_data
from src.feature_engineering import (
    engineer_features,
    validate_no_target_leakage,
    EXCLUDE_FROM_FEATURES,
    TARGET_COLUMN,
)
from src.preprocessing import (
    build_preprocessor,
    prepare_data,
    split_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def convert_isolation_forest_predictions(predictions):
    """Convert IF predictions: -1 (anomaly) → 1, 1 (normal) → 0."""
    return np.where(predictions == -1, 1, 0)


def _print_train_metrics(model_name, y_true, y_pred, zero_division=0):
    """Print accuracy, F1, and classification report."""
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=zero_division)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (anomaly): {f1:.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Anomaly"],
                                zero_division=zero_division))
    return f1


def main():
    set_random_seed()
    setup_directories()

    print_separator("Cloud Cost Anomaly Detection — Model Training")

    print_separator("Step 1: Loading and Preparing Data")

    df = load_data()
    print(f"Raw data shape: {df.shape}")

    df = engineer_features(df)
    print(f"Engineered data shape: {df.shape}")

    X, y = prepare_data(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().to_string()}")
    print(f"Anomaly rate: {y.mean():.2%}")

    validate_no_target_leakage(X.columns.tolist())

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test  set: {X_test.shape[0]} samples")

    # Save test data for evaluate.py
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    X_test.to_csv(os.path.join(processed_dir, "test_features.csv"), index=True)
    y_test.to_csv(os.path.join(processed_dir, "test_labels.csv"), index=True)
    print(f"\nSaved test features → {os.path.join(processed_dir, 'test_features.csv')}")
    print(f"Saved test labels   → {os.path.join(processed_dir, 'test_labels.csv')}")

    # Save feature column names for predict.py
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns, os.path.join(models_dir, "feature_columns.pkl"))
    print(f"Saved feature columns list → {os.path.join(models_dir, 'feature_columns.pkl')}")

    summary = {}

    # Model 1: Majority Baseline
    print_separator("Model 1: Majority Baseline")

    baseline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DummyClassifier(
            strategy="most_frequent",
            random_state=RANDOM_STATE,
        )),
    ])
    baseline.fit(X_train, y_train)

    y_pred_baseline = baseline.predict(X_train)
    summary["Baseline"] = _print_train_metrics(
        "Baseline", y_train, y_pred_baseline, zero_division=0
    )

    joblib.dump(baseline, os.path.join(models_dir, "baseline_model.pkl"))
    print(f"Saved → {os.path.join(models_dir, 'baseline_model.pkl')}")

    # Model 2: Logistic Regression
    print_separator("Model 2: Logistic Regression")

    lr_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])
    lr_pipeline.fit(X_train, y_train)

    y_pred_lr = lr_pipeline.predict(X_train)
    summary["Logistic Regression"] = _print_train_metrics(
        "Logistic Regression", y_train, y_pred_lr
    )

    joblib.dump(lr_pipeline, os.path.join(models_dir, "logistic_regression_model.pkl"))
    print(f"Saved → {os.path.join(models_dir, 'logistic_regression_model.pkl')}")

    # Model 3: Random Forest with GridSearchCV
    print_separator("Model 3: Random Forest (GridSearchCV)")

    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=4,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_train)
    summary["Random Forest"] = _print_train_metrics(
        "Random Forest", y_train, y_pred_rf
    )

    joblib.dump(best_rf, os.path.join(models_dir, "random_forest_model.pkl"))
    print(f"Saved → {os.path.join(models_dir, 'random_forest_model.pkl')}")

    # Model 4: Isolation Forest (unsupervised)
    print_separator("Model 4: Isolation Forest (Unsupervised)")

    # Contamination = proportion of anomalies in training data
    contamination = float(y_train.mean())
    print(f"Contamination (from training data): {contamination:.4f}")

    if_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", IsolationForest(
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_estimators=200,
        )),
    ])

    # Isolation Forest is unsupervised — does NOT use y_train
    if_pipeline.fit(X_train)

    raw_preds = if_pipeline.predict(X_train)
    y_pred_if = convert_isolation_forest_predictions(raw_preds)
    summary["Isolation Forest"] = _print_train_metrics(
        "Isolation Forest", y_train, y_pred_if
    )

    joblib.dump(if_pipeline, os.path.join(models_dir, "isolation_forest_model.pkl"))
    print(f"Saved → {os.path.join(models_dir, 'isolation_forest_model.pkl')}")

    # Summary
    print_separator("Training Summary")

    print(f"{'Model':<25} {'Train F1 (anomaly)':>20}")
    print("-" * 47)
    for model_name, f1_val in summary.items():
        print(f"{model_name:<25} {f1_val:>20.4f}")

    print("\nAll models saved to the models/ directory.")
    print("Run `python src/evaluate.py` to evaluate on the test set.")


if __name__ == "__main__":
    main()
