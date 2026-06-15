"""Loads, validates, and summarises the cloud cost dataset."""

import os
import sys
import pandas as pd

# Allow imports from the project root regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import validate_columns, print_separator

REQUIRED_COLUMNS = [
    'date', 'day_of_week', 'is_weekend', 'month',
    'cloud_provider', 'region', 'service_name', 'environment',
    'usage_amount', 'resource_count', 'data_transfer_gb', 'storage_gb',
    'compute_hours', 'request_count',
    'previous_day_cost', 'rolling_3_day_avg_cost', 'rolling_7_day_avg_cost',
    'cost_usd', 'anomaly_type', 'is_anomaly',
]


def load_data(filepath='data/raw/cloud_cost_data.csv'):
    """Load and validate the cloud cost CSV dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found: {filepath}. "
            "Run: python src/data_generation.py"
        )

    df = pd.read_csv(filepath, parse_dates=['date'])
    validate_columns(df, REQUIRED_COLUMNS)
    return df


def validate_data(df):
    """Run basic data-quality checks and report results."""
    print_separator('Data Validation')
    issues = 0

    neg_cost = (df['cost_usd'] < 0).sum()
    if neg_cost > 0:
        print(f"  [WARN] {neg_cost} records have negative cost_usd")
        issues += 1
    else:
        print("  [OK] No negative cost_usd values.")

    usage_cols = [
        'usage_amount', 'resource_count', 'data_transfer_gb',
        'storage_gb', 'compute_hours', 'request_count',
    ]
    for col in usage_cols:
        neg = (df[col] < 0).sum()
        if neg > 0:
            print(f"  [WARN] {neg} records have negative {col}")
            issues += 1
        else:
            print(f"  [OK] No negative {col} values.")

    anomaly_rate = df['is_anomaly'].mean() * 100
    if 1.0 <= anomaly_rate <= 15.0:
        print(f"  [OK] Anomaly rate {anomaly_rate:.2f}% is within [1%, 15%].")
    else:
        print(f"  [WARN] Anomaly rate {anomaly_rate:.2f}% is outside [1%, 15%].")
        issues += 1

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        print(f"  [WARN] Fully empty columns: {empty_cols}")
        issues += 1
    else:
        print("  [OK] No fully empty columns.")

    if issues == 0:
        print("\n  >>> All validation checks PASSED.")
    else:
        print(f"\n  >>> {issues} validation issue(s) found – review warnings above.")


def print_data_summary(df):
    """Print a concise summary of the dataset to the console."""
    print_separator('Dataset Summary')

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")

    print("\n--- Data Types ---")
    print(df.dtypes.to_string())

    print("\n--- First 5 Rows ---")
    print(df.head().to_string())

    print("\n--- Descriptive Statistics ---")
    print(df.describe().to_string())

    anomaly_count = df['is_anomaly'].sum()
    total = len(df)
    print(f"\nAnomaly count : {anomaly_count} / {total} "
          f"({anomaly_count / total * 100:.2f}%)")

    if 'anomaly_type' in df.columns:
        print("\n--- Anomaly Types ---")
        print(df['anomaly_type'].value_counts().to_string())

    print(f"\nDate range: {df['date'].min()} → {df['date'].max()}")


if __name__ == '__main__':
    df = load_data()
    validate_data(df)
    print_data_summary(df)
