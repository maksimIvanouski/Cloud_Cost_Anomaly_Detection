"""Prediction module for cloud cost anomaly detection."""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import safe_divide, print_separator
from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def create_sample_record(
    cloud_provider,
    region,
    service_name,
    environment,
    usage_amount,
    resource_count,
    data_transfer_gb,
    storage_gb,
    compute_hours,
    request_count,
    cost_usd,
    previous_day_cost,
    rolling_3_day_avg_cost,
    rolling_7_day_avg_cost,
):
    """Create a single-row DataFrame with all engineered features."""
    now = datetime.now()
    day_of_week = now.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    month = now.month
    is_business_day = 0 if is_weekend else 1

    cost_to_rolling_7_day_ratio = safe_divide(cost_usd, rolling_7_day_avg_cost)
    cost_to_rolling_3_day_ratio = safe_divide(cost_usd, rolling_3_day_avg_cost)
    cost_difference_from_previous_day = cost_usd - previous_day_cost
    cost_change_percent = safe_divide(
        cost_difference_from_previous_day, previous_day_cost
    ) * 100

    cost_per_resource = safe_divide(cost_usd, resource_count)
    cost_per_compute_hour = safe_divide(cost_usd, compute_hours)
    cost_per_gb_transfer = safe_divide(cost_usd, data_transfer_gb)
    cost_per_1000_requests = safe_divide(cost_usd * 1000, request_count)
    cost_per_storage_gb = safe_divide(cost_usd, storage_gb)

    # Thresholds chosen from domain logic
    is_high_transfer = 1 if data_transfer_gb > 200 else 0
    is_high_compute = 1 if compute_hours > 500 else 0
    is_high_resource_count = 1 if resource_count > 20 else 0
    is_dev_environment_expensive = (
        1 if environment == "development" and cost_usd > 200 else 0
    )
    is_cost_much_higher_than_average = (
        1 if cost_to_rolling_7_day_ratio > 2.5 else 0
    )
    is_unusual_weekend_activity = (
        1 if is_weekend == 1 and cost_usd > 400 else 0
    )

    record = {
        "cloud_provider": cloud_provider,
        "region": region,
        "service_name": service_name,
        "environment": environment,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "month": month,
        "usage_amount": usage_amount,
        "resource_count": resource_count,
        "data_transfer_gb": data_transfer_gb,
        "storage_gb": storage_gb,
        "compute_hours": compute_hours,
        "request_count": request_count,
        "previous_day_cost": previous_day_cost,
        "rolling_3_day_avg_cost": rolling_3_day_avg_cost,
        "rolling_7_day_avg_cost": rolling_7_day_avg_cost,
        "cost_usd": cost_usd,
        "is_business_day": is_business_day,
        "cost_to_rolling_7_day_ratio": cost_to_rolling_7_day_ratio,
        "cost_to_rolling_3_day_ratio": cost_to_rolling_3_day_ratio,
        "cost_difference_from_previous_day": cost_difference_from_previous_day,
        "cost_change_percent": cost_change_percent,
        "cost_per_resource": cost_per_resource,
        "cost_per_compute_hour": cost_per_compute_hour,
        "cost_per_gb_transfer": cost_per_gb_transfer,
        "cost_per_1000_requests": cost_per_1000_requests,
        "cost_per_storage_gb": cost_per_storage_gb,
        "is_high_transfer": is_high_transfer,
        "is_high_compute": is_high_compute,
        "is_high_resource_count": is_high_resource_count,
        "is_dev_environment_expensive": is_dev_environment_expensive,
        "is_cost_much_higher_than_average": is_cost_much_higher_than_average,
        "is_unusual_weekend_activity": is_unusual_weekend_activity,
    }

    # Enforce the exact column order the pipeline expects
    df = pd.DataFrame([record])
    expected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    return df


def predict_anomaly(record_df, model_path="models/random_forest_model.pkl"):
    """Load a saved pipeline and predict on a single-row DataFrame."""
    pipeline = joblib.load(model_path)
    prediction = pipeline.predict(record_df)

    confidence = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(record_df)
            confidence = proba[0][prediction[0]] * 100
        except Exception:
            pass

    return prediction[0], confidence


def print_suspicious_indicators(record):
    """Return a list of human-readable risk flags for a record."""
    if hasattr(record, "to_dict"):
        record = record.to_dict()

    indicators = []

    if record.get("cost_to_rolling_7_day_ratio", 0) > 2.5:
        indicators.append("cost is much higher than rolling 7-day average")

    if (
        record.get("environment") == "development"
        and record.get("cost_usd", 0) > 200
    ):
        indicators.append("development environment has unusually high cost")

    if record.get("compute_hours", 0) > 500:
        indicators.append("compute hours are high")

    if record.get("data_transfer_gb", 0) > 300:
        indicators.append("data transfer is high")

    if record.get("resource_count", 0) > 25:
        indicators.append("resource count is high")

    if record.get("cost_to_rolling_3_day_ratio", 0) > 3.0:
        indicators.append("cost spike detected vs 3-day average")

    if record.get("is_weekend") == 1 and record.get("cost_usd", 0) > 400:
        indicators.append("high cost on weekend")

    return indicators


def main():
    print_separator("Cloud Cost Anomaly Detection — Prediction Demo")

    model_path = "models/random_forest_model.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        print("Run `python src/train.py` first to train and save models.")
        return

    # Example 1: Normal record
    print_separator("Example 1: Expected NORMAL Record")

    normal_record = create_sample_record(
        cloud_provider="AWS",
        region="us-east-1",
        service_name="EC2",
        environment="production",
        usage_amount=500,
        resource_count=12,
        data_transfer_gb=50,
        storage_gb=200,
        compute_hours=400,
        request_count=50000,
        cost_usd=420,
        previous_day_cost=400,
        rolling_3_day_avg_cost=410,
        rolling_7_day_avg_cost=405,
    )

    prediction, confidence = predict_anomaly(normal_record, model_path)
    record_dict = normal_record.iloc[0].to_dict()
    indicators = print_suspicious_indicators(record_dict)

    print(f"  Provider   : {record_dict['cloud_provider']}")
    print(f"  Service    : EC2")
    print(f"  Environment: {record_dict['environment']}")
    print(f"  Cost       : ${record_dict['cost_usd']:.2f}")
    print(f"  7-day avg  : ${record_dict['rolling_7_day_avg_cost']:.2f}")
    print()
    print(f"  Prediction : {'ANOMALY' if prediction == 1 else 'NORMAL'}")
    if confidence is not None:
        print(f"  Confidence : {confidence:.1f}%")
    if indicators:
        print("  Suspicious indicators:")
        for ind in indicators:
            print(f"    ⚠  {ind}")
    else:
        print("  No suspicious indicators detected.")

    # Example 2: Anomalous record
    print_separator("Example 2: Expected ANOMALY Record")

    anomaly_record = create_sample_record(
        cloud_provider="AWS",
        region="eu-central-1",
        service_name="EC2",
        environment="development",
        usage_amount=850,
        resource_count=40,
        data_transfer_gb=600,
        storage_gb=120,
        compute_hours=700,
        request_count=100000,
        cost_usd=1200,
        previous_day_cost=90,
        rolling_3_day_avg_cost=110,
        rolling_7_day_avg_cost=100,
    )

    prediction, confidence = predict_anomaly(anomaly_record, model_path)
    record_dict = anomaly_record.iloc[0].to_dict()
    indicators = print_suspicious_indicators(record_dict)

    print(f"  Provider   : {record_dict['cloud_provider']}")
    print(f"  Service    : EC2")
    print(f"  Environment: {record_dict['environment']}")
    print(f"  Cost       : ${record_dict['cost_usd']:.2f}")
    print(f"  7-day avg  : ${record_dict['rolling_7_day_avg_cost']:.2f}")
    print()
    print(f"  Prediction : {'ANOMALY' if prediction == 1 else 'NORMAL'}")
    if confidence is not None:
        print(f"  Confidence : {confidence:.1f}%")
    if indicators:
        print("  Suspicious indicators:")
        for ind in indicators:
            print(f"    ⚠  {ind}")
    else:
        print("  No suspicious indicators detected.")

    print_separator("Prediction Demo Complete")


if __name__ == "__main__":
    main()
