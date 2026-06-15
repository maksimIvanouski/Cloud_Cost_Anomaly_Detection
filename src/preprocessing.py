"""Preprocessing pipeline: imputation, scaling, encoding, and train/test split."""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import RANDOM_STATE, print_separator
from src.feature_engineering import (
    EXCLUDE_FROM_FEATURES,
    TARGET_COLUMN,
    validate_no_target_leakage,
)

NUMERIC_FEATURES = [
    'day_of_week', 'is_weekend', 'month',
    'usage_amount', 'resource_count', 'data_transfer_gb', 'storage_gb',
    'compute_hours', 'request_count',
    'previous_day_cost', 'rolling_3_day_avg_cost', 'rolling_7_day_avg_cost',
    'cost_usd',
    'is_business_day',
    'cost_to_rolling_7_day_ratio', 'cost_to_rolling_3_day_ratio',
    'cost_difference_from_previous_day', 'cost_change_percent',
    'cost_per_resource', 'cost_per_compute_hour', 'cost_per_gb_transfer',
    'cost_per_1000_requests', 'cost_per_storage_gb',
    'is_high_transfer', 'is_high_compute', 'is_high_resource_count',
    'is_dev_environment_expensive', 'is_cost_much_higher_than_average',
    'is_unusual_weekend_activity',
]

CATEGORICAL_FEATURES = [
    'cloud_provider', 'region', 'service_name', 'environment',
]


def build_preprocessor():
    """Build and return a ColumnTransformer with numeric and categorical sub-pipelines."""
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, NUMERIC_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def prepare_data(df):
    """Extract feature matrix X and target vector y, with leakage validation."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    validate_no_target_leakage(feature_cols)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    return X, y


def split_data(X, y, test_size=0.2):
    """Perform a stratified train/test split."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def validate_preprocessing_data(df):
    """Run validation checks on the feature-engineered dataframe."""
    print_separator('Preprocessing Data Validation')
    issues = 0

    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [FAIL] Missing columns: {missing}")
        issues += 1
    else:
        print("  [OK] All required columns present.")

    if 'cost_usd' in df.columns:
        neg = (df['cost_usd'] < 0).sum()
        if neg > 0:
            print(f"  [WARN] {neg} records with negative cost_usd")
            issues += 1
        else:
            print("  [OK] No negative cost_usd.")

    usage_cols = [
        'usage_amount', 'resource_count', 'data_transfer_gb',
        'storage_gb', 'compute_hours', 'request_count',
    ]
    for col in usage_cols:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                print(f"  [WARN] {neg} records with negative {col}")
                issues += 1

    if issues == 0:
        print("  [OK] No negative usage values.")

    if TARGET_COLUMN in df.columns:
        rate = df[TARGET_COLUMN].mean() * 100
        if 1.0 <= rate <= 15.0:
            print(f"  [OK] Anomaly rate {rate:.2f}% within [1%, 15%].")
        else:
            print(f"  [WARN] Anomaly rate {rate:.2f}% outside [1%, 15%].")
            issues += 1

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    empty = [c for c in feature_cols if c in df.columns and df[c].isna().all()]
    if empty:
        print(f"  [WARN] Fully empty feature columns: {empty}")
        issues += 1
    else:
        print("  [OK] No fully empty feature columns.")

    if TARGET_COLUMN in df.columns:
        print(f"  [OK] Target column '{TARGET_COLUMN}' exists.")
    else:
        print(f"  [FAIL] Target column '{TARGET_COLUMN}' NOT found!")
        issues += 1

    if issues == 0:
        print("\n  >>> All preprocessing validation checks PASSED.")
    else:
        print(f"\n  >>> {issues} issue(s) found – review warnings above.")


if __name__ == '__main__':
    from src.data_loading import load_data
    from src.feature_engineering import engineer_features

    print_separator('Preprocessing Pipeline Demo')

    df = load_data()
    df = engineer_features(df)
    validate_preprocessing_data(df)

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nTrain set : {X_train.shape[0]} samples")
    print(f"Test  set : {X_test.shape[0]} samples")
    print(f"Train anomaly rate: {y_train.mean() * 100:.2f}%")
    print(f"Test  anomaly rate: {y_test.mean() * 100:.2f}%")

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print(f"\nTransformed train shape: {X_train_transformed.shape}")
    print(f"Transformed test  shape: {X_test_transformed.shape}")
    print("\n[OK] Preprocessing pipeline built and tested successfully.")
