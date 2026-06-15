"""Model evaluation pipeline for cloud cost anomaly detection."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import joblib

# Non-interactive backend must be set BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import RANDOM_STATE, setup_directories, save_plot, print_separator
from src.data_loading import load_data
from src.feature_engineering import (
    engineer_features,
    TARGET_COLUMN,
    EXCLUDE_FROM_FEATURES,
)
from src.preprocessing import (
    prepare_data,
    split_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

COLORS = {
    "Baseline": "#95a5a6",
    "Logistic Regression": "#3498db",
    "Random Forest": "#2ecc71",
    "Isolation Forest": "#e74c3c",
}

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def convert_isolation_forest_predictions(predictions):
    """Convert IF predictions: -1 (anomaly) → 1, 1 (normal) → 0."""
    return np.where(predictions == -1, 1, 0)


def _compute_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics as a dictionary."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": None,
        "Avg Precision": None,
    }
    if y_proba is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
            metrics["Avg Precision"] = average_precision_score(y_true, y_proba)
        except ValueError:
            pass
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, plots_dir):
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Normal", "Anomaly"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()

    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    save_plot(fig, fname, directory=plots_dir)
    plt.close(fig)
    print(f"  Saved → {os.path.join(plots_dir, fname)}")


def plot_roc_curves(results, y_test, plots_dir):
    """ROC curves for models with predict_proba (LR & RF)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name in ["Logistic Regression", "Random Forest"]:
        info = results.get(name)
        if info is None or info["proba"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, info["proba"])
        auc_val = info["metrics"].get("ROC-AUC")
        label = f"{name} (AUC = {auc_val:.3f})" if auc_val else name
        ax.plot(fpr, tpr, label=label, color=COLORS[name], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_plot(fig, "roc_curve_comparison.png", directory=plots_dir)
    plt.close(fig)
    print(f"  Saved → {os.path.join(plots_dir, 'roc_curve_comparison.png')}")


def plot_pr_curves(results, y_test, plots_dir):
    """Precision-Recall curves for LR & RF."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name in ["Logistic Regression", "Random Forest"]:
        info = results.get(name)
        if info is None or info["proba"] is None:
            continue
        prec, rec, _ = precision_recall_curve(y_test, info["proba"])
        ap_val = info["metrics"].get("Avg Precision")
        label = f"{name} (AP = {ap_val:.3f})" if ap_val else name
        ax.plot(rec, prec, label=label, color=COLORS[name], linewidth=2)

    prevalence = y_test.mean()
    ax.axhline(y=prevalence, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline prevalence ({prevalence:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison", fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    save_plot(fig, "precision_recall_curve_comparison.png", directory=plots_dir)
    plt.close(fig)
    print(f"  Saved → {os.path.join(plots_dir, 'precision_recall_curve_comparison.png')}")


def plot_feature_importance(rf_pipeline, plots_dir):
    """Top-15 feature importances from Random Forest."""
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        print("  Warning: could not call get_feature_names_out(); using raw feature list.")

    importances = rf_pipeline.named_steps["classifier"].feature_importances_

    # One-hot encoding may expand feature count
    n = min(len(feature_names), len(importances))
    feat_imp = pd.Series(importances[:n], index=feature_names[:n]).sort_values(ascending=False)

    top15 = feat_imp.head(15).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(range(len(top15)), top15.values, color="#2ecc71", edgecolor="#27ae60")
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels([str(n).replace("num__", "").replace("cat__", "") for n in top15.index])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 15 Feature Importances — Random Forest", fontweight="bold")

    for bar_obj in bars:
        width = bar_obj.get_width()
        ax.text(width + 0.002, bar_obj.get_y() + bar_obj.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    save_plot(fig, "feature_importance_random_forest.png", directory=plots_dir)
    plt.close(fig)
    print(f"  Saved → {os.path.join(plots_dir, 'feature_importance_random_forest.png')}")


def plot_model_comparison(results, plots_dir):
    """Grouped bar chart comparing F1 and Recall across models."""
    model_names = list(results.keys())
    f1_scores = [results[m]["metrics"]["F1"] for m in model_names]
    recalls = [results[m]["metrics"]["Recall"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, f1_scores, width, label="F1-Score",
                   color="#3498db", edgecolor="#2980b9")
    bars2 = ax.bar(x + width / 2, recalls, width, label="Recall",
                   color="#e74c3c", edgecolor="#c0392b")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — F1-Score & Recall", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    for bar_obj in bars1:
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.02,
                f"{bar_obj.get_height():.2f}", ha="center", fontsize=9)
    for bar_obj in bars2:
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.02,
                f"{bar_obj.get_height():.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    save_plot(fig, "model_comparison.png", directory=plots_dir)
    plt.close(fig)
    print(f"  Saved → {os.path.join(plots_dir, 'model_comparison.png')}")


def threshold_analysis(y_true, y_proba, model_name="Random Forest"):
    """Compare default (0.5) and lower (0.3) thresholds."""
    print_separator(f"Threshold Tuning — {model_name}")

    for thresh in [0.5, 0.3]:
        y_pred_t = (y_proba >= thresh).astype(int)
        print(f"\n  Threshold = {thresh}")
        print(f"    Precision : {precision_score(y_true, y_pred_t, zero_division=0):.4f}")
        print(f"    Recall    : {recall_score(y_true, y_pred_t, zero_division=0):.4f}")
        print(f"    F1        : {f1_score(y_true, y_pred_t, zero_division=0):.4f}")

    print("\n  Interpretation:")
    print("    Lowering the threshold increases recall (catches more anomalies)")
    print("    at the cost of precision (more false alarms).  Choose based on")
    print("    the business cost of missed anomalies vs. false alerts.")


def error_analysis(X_test, y_test, y_pred, df_full, model_name="Random Forest"):
    """Analyse misclassified records."""
    print_separator(f"Error Analysis — {model_name}")

    fp_mask = (y_pred == 1) & (y_test.values == 0)
    fn_mask = (y_pred == 0) & (y_test.values == 1)

    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    print(f"  False Positives (normal predicted as anomaly): {n_fp}")
    print(f"  False Negatives (anomaly predicted as normal): {n_fn}")

    test_idx = X_test.index
    df_test = df_full.loc[test_idx].copy() if all(i in df_full.index for i in test_idx) else X_test.copy()

    for col in ["service_name", "environment"]:
        if col in df_test.columns:
            print(f"\n  Errors by {col}:")
            error_mask = fp_mask | fn_mask
            if error_mask.sum() > 0:
                error_records = df_test.loc[error_mask]
                counts = error_records[col].value_counts().head(5)
                for val, cnt in counts.items():
                    print(f"    {val}: {cnt}")

    error_indices = X_test.index[fp_mask | fn_mask]
    if len(error_indices) > 0:
        sample_size = min(5, len(error_indices))
        sample_idx = np.random.choice(error_indices, size=sample_size, replace=False)
        print(f"\n  Sample of {sample_size} misclassified records:")

        sample_df = df_test.loc[sample_idx]
        display_cols = [c for c in ["service_name", "environment", "cost_usd",
                                     "rolling_7_day_avg_cost", TARGET_COLUMN]
                        if c in sample_df.columns]
        if display_cols:
            print(sample_df[display_cols].to_string(index=True))
        else:
            print(sample_df.iloc[:, :5].to_string(index=True))
    else:
        print("\n  No misclassified records — perfect predictions!")


def main():
    setup_directories()
    plots_dir = os.path.join("assets", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print_separator("Cloud Cost Anomaly Detection — Model Evaluation")

    # Recreate the EXACT same test split (same random_state)
    print_separator("Step 1: Preparing Test Data")

    df = load_data()
    df = engineer_features(df)
    X, y = prepare_data(df)
    _, X_test, _, y_test = split_data(X, y)

    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Test anomaly rate: {y_test.mean():.2%}")

    # Load models
    print_separator("Step 2: Loading Models")

    models_dir = "models"
    model_files = {
        "Baseline": "baseline_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "Isolation Forest": "isolation_forest_model.pkl",
    }

    models = {}
    for name, fname in model_files.items():
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"  Loaded {name} from {path}")
        else:
            print(f"  WARNING: {path} not found — skipping {name}")

    if not models:
        print("No models found. Run `python src/train.py` first.")
        return

    # Evaluate each model
    print_separator("Step 3: Computing Metrics")

    results = {}

    for name, pipeline in models.items():
        print(f"\n--- {name} ---")

        if name == "Isolation Forest":
            raw = pipeline.predict(X_test)
            y_pred = convert_isolation_forest_predictions(raw)
            y_proba = None  # IF has no predict_proba
        else:
            y_pred = pipeline.predict(X_test)
            try:
                proba = pipeline.predict_proba(X_test)
                y_proba = proba[:, 1]
            except (AttributeError, IndexError):
                y_proba = None

        metrics = _compute_metrics(y_test, y_pred, y_proba)

        print(classification_report(y_test, y_pred,
                                    target_names=["Normal", "Anomaly"],
                                    zero_division=0))

        results[name] = {
            "metrics": metrics,
            "y_pred": y_pred,
            "proba": y_proba,
        }

    # Comparison table
    print_separator("Model Comparison Table")

    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'Avg Prec':>10}"
    print(header)
    print("-" * len(header))
    for name, info in results.items():
        m = info["metrics"]
        roc_str = f"{m['ROC-AUC']:.4f}" if m["ROC-AUC"] is not None else "   N/A"
        ap_str = f"{m['Avg Precision']:.4f}" if m["Avg Precision"] is not None else "   N/A"
        print(f"{name:<25} {m['Accuracy']:>10.4f} {m['Precision']:>10.4f} "
              f"{m['Recall']:>8.4f} {m['F1']:>8.4f} {roc_str:>9} {ap_str:>10}")

    # Generate plots
    print_separator("Step 4: Generating Plots")

    for name, info in results.items():
        plot_confusion_matrix(y_test, info["y_pred"], name, plots_dir)

    plot_roc_curves(results, y_test, plots_dir)
    plot_pr_curves(results, y_test, plots_dir)

    if "Random Forest" in models:
        plot_feature_importance(models["Random Forest"], plots_dir)

    plot_model_comparison(results, plots_dir)

    # Threshold tuning
    best_supervised = "Random Forest" if "Random Forest" in results else "Logistic Regression"
    if best_supervised in results and results[best_supervised]["proba"] is not None:
        threshold_analysis(y_test, results[best_supervised]["proba"], best_supervised)

    # Error analysis
    if best_supervised in results:
        error_analysis(X_test, y_test, results[best_supervised]["y_pred"],
                       df, best_supervised)

    print_separator("Evaluation Complete")
    print(f"All plots saved to {plots_dir}/")
    print("Run `python src/predict.py` to make predictions on new records.")


if __name__ == "__main__":
    main()
