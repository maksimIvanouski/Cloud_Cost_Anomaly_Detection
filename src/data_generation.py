"""Synthetic cloud cost data generator with ~7% injected anomalies."""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import RANDOM_STATE, setup_directories, print_separator

NUM_DAYS = 90
RECORDS_PER_DAY = 56
ANOMALY_FRACTION = 0.07

PROVIDERS = ["AWS", "Azure", "Google Cloud"]
REGIONS = ["us-east-1", "eu-central-1", "eu-west-1", "us-west-2"]
SERVICES = ["EC2", "S3", "RDS", "Lambda", "CloudWatch", "Data Transfer", "Kubernetes"]
ENVIRONMENTS = ["production", "staging", "development"]

# {service: {env: (mean, std)}}
SERVICE_PROFILES = {
    "EC2":           {"production": (400, 120), "staging": (150, 50),  "development": (50, 20)},
    "S3":            {"production": (100, 40),  "staging": (40, 15),   "development": (15, 8)},
    "RDS":           {"production": (300, 90),  "staging": (100, 35),  "development": (40, 15)},
    "Lambda":        {"production": (60, 25),   "staging": (25, 10),   "development": (10, 5)},
    "CloudWatch":    {"production": (30, 12),   "staging": (12, 5),    "development": (5, 3)},
    "Data Transfer": {"production": (80, 35),   "staging": (30, 12),   "development": (12, 6)},
    "Kubernetes":    {"production": (500, 150),  "staging": (180, 60),  "development": (70, 25)},
}

# (usage_mean, resource_mean, transfer_mean, storage_mean, compute_mean, request_mean)
USAGE_PROFILES = {
    "EC2":           (500, 10, 40,  150, 350, 40000),
    "S3":            (300,  5, 80,  500,  10, 80000),
    "RDS":           (200,  4, 20,  300,  50, 20000),
    "Lambda":        (800,  3,  5,   10,  30, 150000),
    "CloudWatch":    (400,  2, 10,   20,  15, 60000),
    "Data Transfer": (600,  3, 120,  30,  20, 70000),
    "Kubernetes":    (450, 12, 60,  200, 400, 50000),
}

ANOMALY_TYPES = [
    "sudden_compute_spike",
    "abnormal_data_transfer",
    "storage_growth",
    "forgotten_test_environment",
    "unexpected_region_usage",
    "monitoring_cost_spike",
    "weekend_production_activity",
    "duplicated_resource_deployment",
]


def _generate_base_records(rng, start_date):
    """Generate normal daily cost records for all combos."""
    rows = []
    all_combos = [
        (p, r, s, e)
        for p in PROVIDERS
        for r in REGIONS
        for s in SERVICES
        for e in ENVIRONMENTS
    ]

    for day_offset in range(NUM_DAYS):
        current_date = start_date + timedelta(days=day_offset)
        day_of_week = current_date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        month = current_date.month

        n_combos = min(RECORDS_PER_DAY, len(all_combos))
        indices = rng.choice(len(all_combos), size=n_combos, replace=False)

        for idx in indices:
            provider, region, service, env = all_combos[idx]
            mean_cost, std_cost = SERVICE_PROFILES[service][env]
            u_mean, r_mean, t_mean, s_mean, c_mean, req_mean = USAGE_PROFILES[service]

            env_mult = {"production": 1.0, "staging": 0.4, "development": 0.15}[env]
            wknd_mult = 0.85 if (is_weekend and env == "production") else (0.6 if is_weekend else 1.0)

            overall_mult = env_mult * wknd_mult

            usage_amount  = max(1, rng.normal(u_mean * overall_mult, u_mean * 0.2))
            resource_count = max(1, int(rng.normal(r_mean * env_mult, r_mean * 0.3)))
            data_transfer  = max(0.1, rng.normal(t_mean * overall_mult, t_mean * 0.25))
            storage        = max(0.1, rng.normal(s_mean * env_mult, s_mean * 0.2))
            compute_hours  = max(0.1, rng.normal(c_mean * overall_mult, c_mean * 0.25))
            request_count  = max(10, int(rng.normal(req_mean * overall_mult, req_mean * 0.2)))

            base_cost = rng.normal(mean_cost, std_cost)
            usage_factor = 1.0 + 0.05 * (usage_amount / (u_mean * overall_mult + 1) - 1.0)
            cost_usd = max(1.0, base_cost * usage_factor)

            rows.append({
                "date": current_date,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "month": month,
                "cloud_provider": provider,
                "region": region,
                "service_name": service,
                "environment": env,
                "usage_amount": round(usage_amount, 2),
                "resource_count": resource_count,
                "data_transfer_gb": round(data_transfer, 2),
                "storage_gb": round(storage, 2),
                "compute_hours": round(compute_hours, 2),
                "request_count": request_count,
                "cost_usd": round(cost_usd, 2),
                "anomaly_type": "normal",
                "is_anomaly": 0,
            })

    return pd.DataFrame(rows)


def _add_rolling_features(df):
    """Compute temporal rolling features using only past values (no leakage)."""
    df = df.sort_values("date").reset_index(drop=True)

    group_cols = ["cloud_provider", "region", "service_name", "environment"]

    df["previous_day_cost"] = df.groupby(group_cols)["cost_usd"].shift(1)
    df["rolling_3_day_avg_cost"] = (
        df.groupby(group_cols)["cost_usd"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df["rolling_7_day_avg_cost"] = (
        df.groupby(group_cols)["cost_usd"]
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    )

    # Fill NaN for first records with their own cost
    df["previous_day_cost"] = df["previous_day_cost"].fillna(df["cost_usd"])
    df["rolling_3_day_avg_cost"] = df["rolling_3_day_avg_cost"].fillna(df["cost_usd"])
    df["rolling_7_day_avg_cost"] = df["rolling_7_day_avg_cost"].fillna(df["cost_usd"])

    for col in ["previous_day_cost", "rolling_3_day_avg_cost", "rolling_7_day_avg_cost"]:
        df[col] = df[col].round(2)

    return df


def _inject_anomalies(df, rng):
    """Inject ~7% anomalies of 8 different types."""
    n_anomalies = int(len(df) * ANOMALY_FRACTION)
    per_type = n_anomalies // len(ANOMALY_TYPES)

    available = df[df["is_anomaly"] == 0].index.tolist()
    rng.shuffle(available)

    ptr = 0

    for atype in ANOMALY_TYPES:
        count = per_type
        selected = available[ptr:ptr + count]
        ptr += count

        for idx in selected:
            row = df.loc[idx]
            if atype == "sudden_compute_spike":
                mult = rng.uniform(4, 8)
                df.at[idx, "cost_usd"] = round(row["cost_usd"] * mult, 2)
                df.at[idx, "compute_hours"] = round(row["compute_hours"] * rng.uniform(5, 10), 2)
                df.at[idx, "resource_count"] = int(row["resource_count"] * rng.uniform(3, 5))

            elif atype == "abnormal_data_transfer":
                mult = rng.uniform(3, 6)
                df.at[idx, "cost_usd"] = round(row["cost_usd"] * mult, 2)
                df.at[idx, "data_transfer_gb"] = round(row["data_transfer_gb"] * rng.uniform(10, 20), 2)

            elif atype == "storage_growth":
                mult = rng.uniform(2, 4)
                df.at[idx, "cost_usd"] = round(row["cost_usd"] * mult, 2)
                df.at[idx, "storage_gb"] = round(row["storage_gb"] * rng.uniform(8, 15), 2)

            elif atype == "forgotten_test_environment":
                # Higher multiplier for dev/staging where this anomaly is more realistic
                if row["environment"] in ("development", "staging"):
                    mult = rng.uniform(5, 10)
                else:
                    mult = rng.uniform(2, 4)
                df.at[idx, "cost_usd"] = round(row["cost_usd"] * mult, 2)

            elif atype == "unexpected_region_usage":
                mult = rng.uniform(3, 5)
                df.at[idx, "cost_usd"] = round(row["cost_usd"] * mult, 2)

            elif atype == "monitoring_cost_spike":
                mult = rng.uniform(5, 10)
                df.at[idx, "cost_usd"] = round(row["cost_usd"] * mult, 2)
                df.at[idx, "request_count"] = int(row["request_count"] * rng.uniform(3, 8))

            elif atype == "weekend_production_activity":
                mult = rng.uniform(3, 5)
                df.at[idx, "cost_usd"] = round(row["cost_usd"] * mult, 2)
                df.at[idx, "compute_hours"] = round(row["compute_hours"] * rng.uniform(2, 4), 2)
                df.at[idx, "usage_amount"] = round(row["usage_amount"] * rng.uniform(2, 3), 2)

            elif atype == "duplicated_resource_deployment":
                mult = rng.uniform(3, 5)
                df.at[idx, "cost_usd"] = round(row["cost_usd"] * mult, 2)
                df.at[idx, "resource_count"] = int(row["resource_count"] * rng.uniform(4, 8))

            df.at[idx, "anomaly_type"] = atype
            df.at[idx, "is_anomaly"] = 1

    return df


def generate_cloud_cost_data(random_state=RANDOM_STATE):
    """Generate the full synthetic cloud cost dataset with anomalies."""
    rng = np.random.RandomState(random_state)
    start_date = datetime(2025, 1, 1)

    print_separator("Generating Synthetic Cloud Cost Data")
    print(f"  Random state : {random_state}")
    print(f"  Date range   : {start_date.date()} -> "
          f"{(start_date + timedelta(days=NUM_DAYS - 1)).date()}")

    df = _generate_base_records(rng, start_date)
    print(f"  Base records : {len(df)}")

    # Rolling features computed BEFORE anomalies so averages reflect normal behaviour
    df = _add_rolling_features(df)
    print("  Rolling features computed (past values only).")

    df = _inject_anomalies(df, rng)
    n_anomalies = df["is_anomaly"].sum()
    print(f"  Anomalies    : {n_anomalies} / {len(df)} "
          f"({n_anomalies / len(df) * 100:.1f}%)")

    column_order = [
        "date", "day_of_week", "is_weekend", "month",
        "cloud_provider", "region", "service_name", "environment",
        "usage_amount", "resource_count", "data_transfer_gb", "storage_gb",
        "compute_hours", "request_count",
        "previous_day_cost", "rolling_3_day_avg_cost", "rolling_7_day_avg_cost",
        "cost_usd", "anomaly_type", "is_anomaly",
    ]
    df = df[column_order]

    return df


def main():
    setup_directories()

    df = generate_cloud_cost_data()

    output_path = os.path.join("data", "raw", "cloud_cost_data.csv")
    df.to_csv(output_path, index=False)
    print(f"\n  Saved -> {output_path}")

    print_separator("Dataset Summary")
    print(f"  Total records    : {len(df)}")
    print(f"  Columns          : {df.shape[1]}")
    print(f"  Date range       : {df['date'].min()} -> {df['date'].max()}")
    print(f"  Anomaly count    : {df['is_anomaly'].sum()}")
    print(f"  Anomaly rate     : {df['is_anomaly'].mean() * 100:.1f}%")

    print("\n  Records by service:")
    for svc, cnt in df["service_name"].value_counts().items():
        print(f"    {svc:20s}  {cnt}")

    print("\n  Records by environment:")
    for env, cnt in df["environment"].value_counts().items():
        print(f"    {env:20s}  {cnt}")

    print("\n  Anomaly types:")
    atype_counts = df[df["is_anomaly"] == 1]["anomaly_type"].value_counts()
    for at, cnt in atype_counts.items():
        print(f"    {at:40s}  {cnt}")

    print(f"\n  Cost range: ${df['cost_usd'].min():.2f} – ${df['cost_usd'].max():.2f}")
    print(f"  Mean cost (normal)  : ${df[df['is_anomaly'] == 0]['cost_usd'].mean():.2f}")
    print(f"  Mean cost (anomaly) : ${df[df['is_anomaly'] == 1]['cost_usd'].mean():.2f}")
    print("\n  Done. Run `python src/exploratory_analysis.py` next.")


if __name__ == "__main__":
    main()
