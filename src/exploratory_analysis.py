"""Generates EDA plots from the cloud cost dataset and saves them to assets/plots/."""

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import setup_directories, save_plot, print_separator
from src.data_loading import load_data

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    try:
        plt.style.use("seaborn-darkgrid")
    except OSError:
        pass  # fall back to default

COLOR_NORMAL = "#2ecc71"
COLOR_ANOMALY = "#e74c3c"
COLOR_PRIMARY = "#3498db"
COLOR_SECONDARY = "#9b59b6"
COLOR_ACCENT = "#f39c12"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
})


def plot_class_distribution(df, plots_dir):
    """Bar chart of Normal vs Anomaly counts with percentages."""
    counts = df["is_anomaly"].value_counts().sort_index()
    labels = ["Normal (0)", "Anomaly (1)"]
    colors = [COLOR_NORMAL, COLOR_ANOMALY]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)

    total = len(df)
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                f"{val:,}\n({pct:.1f}%)", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Number of Records")
    ax.set_title("Class Distribution — Normal vs Anomaly", fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    save_plot(fig, "class_distribution.png", directory=plots_dir)


def plot_cost_over_time(df, plots_dir):
    """Line chart of daily total cost with anomaly points highlighted."""
    df_time = df.copy()
    df_time["date"] = pd.to_datetime(df_time["date"])

    daily_total = df_time.groupby("date")["cost_usd"].sum().reset_index()
    daily_anomaly = (
        df_time[df_time["is_anomaly"] == 1]
        .groupby("date")["cost_usd"].sum().reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_total["date"], daily_total["cost_usd"],
            color=COLOR_PRIMARY, linewidth=1.3, label="Total daily cost")

    if not daily_anomaly.empty:
        ax.scatter(daily_anomaly["date"], daily_anomaly["cost_usd"],
                   color=COLOR_ANOMALY, s=30, zorder=5, label="Anomaly cost", alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Cloud Cost Over Time with Anomaly Highlights", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_plot(fig, "cost_over_time.png", directory=plots_dir)


def plot_cost_distribution(df, plots_dir):
    """Histogram of cost_usd, full and split by class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["cost_usd"], bins=60, color=COLOR_PRIMARY, edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Cost (USD)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Cost Distribution — All Records", fontweight="bold")

    normal = df[df["is_anomaly"] == 0]["cost_usd"]
    anomaly = df[df["is_anomaly"] == 1]["cost_usd"]
    axes[1].hist(normal, bins=50, color=COLOR_NORMAL, edgecolor="white",
                 alpha=0.7, label="Normal")
    axes[1].hist(anomaly, bins=30, color=COLOR_ANOMALY, edgecolor="white",
                 alpha=0.7, label="Anomaly")
    axes[1].set_xlabel("Cost (USD)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Cost Distribution — Normal vs Anomaly", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    save_plot(fig, "cost_distribution.png", directory=plots_dir)


def plot_cost_by_service(df, plots_dir):
    """Horizontal bar chart of average cost per service."""
    svc_stats = df.groupby("service_name").agg(
        avg_cost=("cost_usd", "mean"),
        total_cost=("cost_usd", "sum"),
    ).sort_values("avg_cost", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(svc_stats.index, svc_stats["avg_cost"],
                   color=COLOR_PRIMARY, edgecolor="#2980b9")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 2, bar.get_y() + bar.get_height() / 2,
                f"${w:,.0f}", va="center", fontsize=10)

    ax.set_xlabel("Average Cost (USD)")
    ax.set_title("Average Cost by Cloud Service", fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "cost_by_service.png", directory=plots_dir)


def plot_cost_by_provider(df, plots_dir):
    """Bar chart of average cost per cloud provider."""
    prov = df.groupby("cloud_provider")["cost_usd"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT][:len(prov)]
    bars = ax.bar(prov.index, prov.values, color=colors, edgecolor="white", width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f"${h:,.0f}", ha="center", fontsize=11)

    ax.set_ylabel("Average Cost (USD)")
    ax.set_title("Average Cost by Cloud Provider", fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "cost_by_provider.png", directory=plots_dir)


def plot_cost_by_environment(df, plots_dir):
    """Bar chart of average cost per environment."""
    env = df.groupby("environment")["cost_usd"].mean().reindex(
        ["production", "staging", "development"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax.bar(env.index, env.values, color=colors, edgecolor="white", width=0.45)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"${h:,.0f}", ha="center", fontsize=11)

    ax.set_ylabel("Average Cost (USD)")
    ax.set_title("Average Cost by Environment", fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "cost_by_environment.png", directory=plots_dir)


def plot_anomaly_type_distribution(df, plots_dir):
    """Horizontal bar chart of anomaly type counts."""
    anomalies = df[df["is_anomaly"] == 1]
    if anomalies.empty:
        return

    atype = anomalies["anomaly_type"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(atype.index, atype.values, color=COLOR_ANOMALY, edgecolor="#c0392b")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                str(int(w)), va="center", fontsize=10)

    ax.set_xlabel("Count")
    ax.set_title("Anomaly Type Distribution", fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "anomaly_type_distribution.png", directory=plots_dir)


def plot_cost_by_region(df, plots_dir):
    """Bar chart of average cost per region."""
    reg = df.groupby("region")["cost_usd"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(reg.index, reg.values, color=COLOR_PRIMARY, edgecolor="#2980b9", width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"${h:,.0f}", ha="center", fontsize=10)

    ax.set_ylabel("Average Cost (USD)")
    ax.set_title("Average Cost by Region", fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "cost_by_region.png", directory=plots_dir)


def plot_rolling_avg_vs_actual(df, plots_dir):
    """Actual cost vs rolling 7-day average for a sample service/env."""
    df_t = df.copy()
    df_t["date"] = pd.to_datetime(df_t["date"])

    # Pick a common combination for a clear view
    mask = (
        (df_t["service_name"] == "EC2")
        & (df_t["environment"] == "production")
        & (df_t["cloud_provider"] == "AWS")
        & (df_t["region"] == "us-east-1")
    )
    subset = df_t[mask].sort_values("date")

    if subset.empty:
        # Fallback: use the largest group if preferred combo is missing
        combo = df_t.groupby(["service_name", "environment"]).size().idxmax()
        mask = (df_t["service_name"] == combo[0]) & (df_t["environment"] == combo[1])
        subset = df_t[mask].sort_values("date")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(subset["date"], subset["cost_usd"],
            color=COLOR_PRIMARY, linewidth=1.2, label="Actual cost", alpha=0.9)
    ax.plot(subset["date"], subset["rolling_7_day_avg_cost"],
            color=COLOR_ACCENT, linewidth=1.5, linestyle="--", label="Rolling 7-day avg")

    anom = subset[subset["is_anomaly"] == 1]
    if not anom.empty:
        ax.scatter(anom["date"], anom["cost_usd"],
                   color=COLOR_ANOMALY, s=60, zorder=5, label="Anomaly", edgecolors="darkred")

    svc = subset["service_name"].iloc[0]
    env = subset["environment"].iloc[0]
    ax.set_xlabel("Date")
    ax.set_ylabel("Cost (USD)")
    ax.set_title(f"Actual Cost vs Rolling Average — {svc} ({env})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_plot(fig, "rolling_avg_vs_actual.png", directory=plots_dir)


def plot_correlation_heatmap(df, plots_dir):
    """Heatmap of numeric feature correlations."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target to avoid misleading correlations
    numeric_cols = [c for c in numeric_cols if c != "is_anomaly"]

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(numeric_cols, fontsize=8)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")

    plt.tight_layout()
    save_plot(fig, "correlation_heatmap.png", directory=plots_dir)


def main():
    setup_directories()
    plots_dir = os.path.join("assets", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print_separator("Exploratory Data Analysis")

    df = load_data()

    print(f"\n  Records        : {len(df):,}")
    print(f"  Features       : {df.shape[1]}")
    n_anom = int(df["is_anomaly"].sum())
    n_norm = len(df) - n_anom
    print(f"  Normal records : {n_norm:,}")
    print(f"  Anomaly records: {n_anom:,}")
    print(f"  Anomaly rate   : {n_anom / len(df) * 100:.1f}%")
    print(f"  Date range     : {df['date'].min()} → {df['date'].max()}")

    print(f"\n  Generating plots → {plots_dir}/\n")

    plot_class_distribution(df, plots_dir)
    plot_cost_over_time(df, plots_dir)
    plot_cost_distribution(df, plots_dir)
    plot_cost_by_service(df, plots_dir)
    plot_cost_by_provider(df, plots_dir)
    plot_cost_by_environment(df, plots_dir)
    plot_anomaly_type_distribution(df, plots_dir)
    plot_cost_by_region(df, plots_dir)
    plot_rolling_avg_vs_actual(df, plots_dir)
    plot_correlation_heatmap(df, plots_dir)

    print(f"\n  ✓ All 10 plots saved to {plots_dir}/")
    print("  Run `python src/train.py` next.")


if __name__ == "__main__":
    main()
