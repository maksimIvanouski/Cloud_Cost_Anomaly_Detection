"""Transforms raw cloud cost data into ML-ready features."""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import safe_divide, print_separator, setup_directories
from src.data_loading import load_data

EXCLUDE_FROM_FEATURES = ['is_anomaly', 'anomaly_type', 'date']
TARGET_COLUMN = 'is_anomaly'


def engineer_features(df):
    """Create all derived features and return an enriched DataFrame."""
    df = df.copy()

    df['is_business_day'] = (df['is_weekend'] == 0).astype(int)

    df['cost_to_rolling_7_day_ratio'] = safe_divide(
        df['cost_usd'], df['rolling_7_day_avg_cost']
    )
    df['cost_to_rolling_3_day_ratio'] = safe_divide(
        df['cost_usd'], df['rolling_3_day_avg_cost']
    )
    df['cost_difference_from_previous_day'] = (
        df['cost_usd'] - df['previous_day_cost']
    )
    df['cost_change_percent'] = (
        safe_divide(df['cost_difference_from_previous_day'],
                    df['previous_day_cost']) * 100
    )

    df['cost_per_resource'] = safe_divide(
        df['cost_usd'], df['resource_count']
    )
    df['cost_per_compute_hour'] = safe_divide(
        df['cost_usd'], df['compute_hours']
    )
    df['cost_per_gb_transfer'] = safe_divide(
        df['cost_usd'], df['data_transfer_gb']
    )
    df['cost_per_1000_requests'] = safe_divide(
        df['cost_usd'] * 1000, df['request_count']
    )
    df['cost_per_storage_gb'] = safe_divide(
        df['cost_usd'], df['storage_gb']
    )

    # Rule-based flags — derived from input data only (no target leakage)
    df['is_high_transfer'] = (
        df['data_transfer_gb'] > df['data_transfer_gb'].quantile(0.90)
    ).astype(int)

    df['is_high_compute'] = (
        df['compute_hours'] > df['compute_hours'].quantile(0.90)
    ).astype(int)

    df['is_high_resource_count'] = (
        df['resource_count'] > df['resource_count'].quantile(0.90)
    ).astype(int)

    df['is_dev_environment_expensive'] = (
        (df['environment'] == 'development')
        & (df['cost_usd'] > df['cost_usd'].quantile(0.75))
    ).astype(int)

    df['is_cost_much_higher_than_average'] = (
        df['cost_to_rolling_7_day_ratio'] > 3.0
    ).astype(int)

    df['is_unusual_weekend_activity'] = (
        (df['is_weekend'] == 1)
        & (df['cost_usd'] > df['cost_usd'].quantile(0.80))
    ).astype(int)

    return df


def get_feature_columns(df):
    """Return column names to use as ML features."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    return feature_cols


def validate_no_target_leakage(feature_columns):
    """Raise ValueError if target-related columns appear in features."""
    forbidden = {'is_anomaly', 'anomaly_type'}
    leaked = forbidden.intersection(set(feature_columns))
    if leaked:
        raise ValueError(
            f"TARGET LEAKAGE DETECTED! Columns {leaked} found in features!"
        )
    print("[OK] No target leakage detected in feature columns.")


if __name__ == '__main__':
    print_separator('Feature Engineering')

    df = load_data()
    df = engineer_features(df)

    feature_cols = get_feature_columns(df)
    validate_no_target_leakage(feature_cols)

    setup_directories()
    output_path = 'data/processed/cloud_cost_features.csv'
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} records with {len(feature_cols)} features "
          f"to {output_path}")
    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
