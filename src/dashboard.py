"""Optional Streamlit dashboard for cloud cost anomaly detection."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import streamlit as st
except ImportError:
    print("=" * 60)
    print("Streamlit is not installed.")
    print("Install with:  pip install streamlit")
    print()
    print("The dashboard is optional.  Core functionality works via")
    print("command-line scripts (train.py, evaluate.py, predict.py).")
    print("=" * 60)
    sys.exit(0)

import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import safe_divide
from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.predict import create_sample_record, predict_anomaly, print_suspicious_indicators


st.set_page_config(
    page_title="Cloud Cost Anomaly Detection",
    page_icon="☁️",
    layout="wide",
)


@st.cache_data
def load_raw_data():
    """Load the raw CSV dataset."""
    path = os.path.join("data", "raw", "cloud_cost_data.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model(model_path):
    """Load a saved sklearn pipeline."""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


st.sidebar.title("☁️ Cloud Cost Anomaly Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Dataset Overview", "Cost Analysis", "Model Results", "Predict"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard is **optional**.  The ML pipeline runs entirely "
    "from the command line via `train.py`, `evaluate.py`, and `predict.py`."
)


if page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("Explore the raw cloud cost dataset used for training and evaluation.")

    df = load_raw_data()
    if df is None:
        st.error("Dataset not found at `data/raw/cloud_cost_data.csv`.  "
                 "Run `python src/generate_data.py` first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Features", f"{df.shape[1]}")

        if "is_anomaly" in df.columns:
            n_anomalies = int(df["is_anomaly"].sum())
            pct = n_anomalies / len(df) * 100
            col3.metric("Anomalies", f"{n_anomalies:,}")
            col4.metric("Anomaly Rate", f"{pct:.1f}%")
        else:
            col3.metric("Anomalies", "N/A")
            col4.metric("Anomaly Rate", "N/A")

        st.subheader("Sample Rows")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Column Types")
        st.write(df.dtypes.to_frame("dtype"))

        st.subheader("Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)


elif page == "Cost Analysis":
    st.title("💰 Cost Analysis")

    df = load_raw_data()
    if df is None:
        st.error("Dataset not found.  Run `python src/generate_data.py` first.")
    else:
        if "date" in df.columns and "cost_usd" in df.columns:
            st.subheader("Cost Over Time")
            df_time = df.copy()
            df_time["date"] = pd.to_datetime(df_time["date"], errors="coerce")
            daily = df_time.groupby("date")["cost_usd"].sum().reset_index()
            daily = daily.sort_values("date")

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(daily["date"], daily["cost_usd"], color="#3498db", linewidth=1.2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Total Daily Cost (USD)")
            ax.set_title("Total Daily Cloud Cost")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if "service_name" in df.columns and "cost_usd" in df.columns:
            st.subheader("Cost by Service")
            svc_cost = df.groupby("service_name")["cost_usd"].sum().sort_values(ascending=False)

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            bars = ax2.bar(svc_cost.index, svc_cost.values, color="#2ecc71", edgecolor="#27ae60")
            ax2.set_xlabel("Service")
            ax2.set_ylabel("Total Cost (USD)")
            ax2.set_title("Total Cost by Cloud Service")
            ax2.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        if "is_anomaly" in df.columns:
            st.subheader("Anomaly Distribution")
            col1, col2 = st.columns(2)

            with col1:
                counts = df["is_anomaly"].value_counts()
                fig3, ax3 = plt.subplots(figsize=(5, 5))
                ax3.pie(counts.values,
                        labels=["Normal", "Anomaly"],
                        autopct="%1.1f%%",
                        colors=["#2ecc71", "#e74c3c"],
                        startangle=90)
                ax3.set_title("Normal vs Anomaly")
                st.pyplot(fig3)
                plt.close(fig3)

            with col2:
                if "anomaly_type" in df.columns:
                    atype = df[df["is_anomaly"] == 1]["anomaly_type"].value_counts()
                    fig4, ax4 = plt.subplots(figsize=(6, 5))
                    ax4.barh(atype.index, atype.values, color="#e74c3c", edgecolor="#c0392b")
                    ax4.set_xlabel("Count")
                    ax4.set_title("Anomaly Types")
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close(fig4)


elif page == "Model Results":
    st.title("🏆 Model Results")
    st.markdown("Comparison of all trained models on the test set.")

    st.subheader("Confusion Matrices")
    plots_dir = os.path.join("assets", "plots")

    cm_files = {
        "Baseline": "confusion_matrix_baseline.png",
        "Logistic Regression": "confusion_matrix_logistic_regression.png",
        "Random Forest": "confusion_matrix_random_forest.png",
        "Isolation Forest": "confusion_matrix_isolation_forest.png",
    }

    cols = st.columns(2)
    for idx, (name, fname) in enumerate(cm_files.items()):
        path = os.path.join(plots_dir, fname)
        col = cols[idx % 2]
        with col:
            if os.path.exists(path):
                st.image(path, caption=name, use_container_width=True)
            else:
                st.warning(f"{fname} not found.  Run `python src/evaluate.py`.")

    st.subheader("Comparison Charts")
    comparison_plots = [
        ("ROC Curve Comparison", "roc_curve_comparison.png"),
        ("Precision-Recall Curve", "precision_recall_curve_comparison.png"),
        ("Feature Importance (RF)", "feature_importance_random_forest.png"),
        ("Model Comparison", "model_comparison.png"),
    ]

    cols2 = st.columns(2)
    for idx, (title, fname) in enumerate(comparison_plots):
        path = os.path.join(plots_dir, fname)
        col = cols2[idx % 2]
        with col:
            if os.path.exists(path):
                st.image(path, caption=title, use_container_width=True)
            else:
                st.warning(f"{fname} not found.")


elif page == "Predict":
    st.title("🔮 Predict Anomaly")
    st.markdown("Enter cloud cost details to check if a record is anomalous.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            cloud_provider = st.selectbox("Cloud Provider", ["AWS", "Azure", "GCP"])
            region = st.selectbox("Region", [
                "us-east-1", "us-west-2", "eu-west-1", "eu-central-1",
                "ap-southeast-1", "ap-northeast-1",
            ])
            service_name = st.selectbox("Service", [
                "EC2", "S3", "RDS", "Lambda", "CloudFront",
                "DynamoDB", "EKS", "ElastiCache",
            ])
            environment = st.selectbox("Environment", [
                "production", "staging", "development", "testing",
            ])

        with col2:
            usage_amount = st.number_input("Usage Amount", value=500.0, step=50.0)
            resource_count = st.number_input("Resource Count", value=12, step=1)
            data_transfer_gb = st.number_input("Data Transfer (GB)", value=50.0, step=10.0)
            storage_gb = st.number_input("Storage (GB)", value=200.0, step=10.0)
            compute_hours = st.number_input("Compute Hours", value=400.0, step=50.0)

        with col3:
            request_count = st.number_input("Request Count", value=50000, step=5000)
            cost_usd = st.number_input("Cost (USD)", value=420.0, step=10.0)
            previous_day_cost = st.number_input("Previous Day Cost (USD)", value=400.0, step=10.0)
            rolling_3_day_avg = st.number_input("Rolling 3-Day Avg Cost", value=410.0, step=10.0)
            rolling_7_day_avg = st.number_input("Rolling 7-Day Avg Cost", value=405.0, step=10.0)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        record_df = create_sample_record(
            cloud_provider=cloud_provider,
            region=region,
            service_name=service_name,
            environment=environment,
            usage_amount=usage_amount,
            resource_count=resource_count,
            data_transfer_gb=data_transfer_gb,
            storage_gb=storage_gb,
            compute_hours=compute_hours,
            request_count=request_count,
            cost_usd=cost_usd,
            previous_day_cost=previous_day_cost,
            rolling_3_day_avg_cost=rolling_3_day_avg,
            rolling_7_day_avg_cost=rolling_7_day_avg,
        )

        model_path = "models/random_forest_model.pkl"
        if not os.path.exists(model_path):
            st.error("Model not found.  Run `python src/train.py` first.")
        else:
            prediction, confidence = predict_anomaly(record_df, model_path)
            record_dict = record_df.iloc[0].to_dict()
            indicators = print_suspicious_indicators(record_dict)

            st.markdown("---")

            if prediction == 1:
                st.error("🚨 **ANOMALY DETECTED**")
            else:
                st.success("✅ **NORMAL** — No anomaly detected")

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Prediction", "ANOMALY" if prediction == 1 else "NORMAL")
            if confidence is not None:
                mcol2.metric("Confidence", f"{confidence:.1f}%")
            mcol3.metric("Cost vs 7-day Avg",
                         f"{record_dict.get('cost_to_rolling_7_day_ratio', 0):.2f}x")

            if indicators:
                st.subheader("⚠️ Suspicious Indicators")
                for ind in indicators:
                    st.warning(f"• {ind}")
            else:
                st.info("No suspicious indicators detected.")
