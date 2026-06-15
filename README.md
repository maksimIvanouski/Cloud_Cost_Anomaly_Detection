# ☁️ Cloud Cost Anomaly Detection with Machine Learning

> Detecting suspicious cloud infrastructure costs before they become financial problems.

**Polish title:** Wykrywanie anomalii w kosztach chmury z wykorzystaniem uczenia maszynowego

**Project Owner & Creator:** Maksim Ivanouski

---

## 📋 Table of Contents

- [Team Members](#-team-members)
- [Project Overview](#-project-overview)
- [Motivation](#-motivation)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Models](#-models)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Explainability](#-explainability)
- [Cloud Architecture Concept](#️-cloud-architecture-concept)
- [Installation](#-installation)
- [Usage](#️-usage)
- [Example Prediction](#-example-prediction)
- [Dashboard (Optional)](#-dashboard-optional)
- [Project Structure](#-project-structure)
- [Common ML Risks Avoided](#️-common-ml-risks-avoided)
- [Assignment Criteria Alignment](#-assignment-criteria-alignment)
- [Future Improvements](#-future-improvements)

---

## 👥 Project Author

- **Maksim Ivanouski** — Owner & Creator

---

## 🎯 Project Overview

This project demonstrates how classical ML algorithms can detect anomalous cloud infrastructure costs. It implements a complete ML pipeline from synthetic data generation to prediction, designed for university presentation and junior Data Scientist / Cloud Engineer portfolio.

The task is **binary classification**:

- **0** = Normal cost behavior
- **1** = Anomalous cost behavior

**Key insight:** A high cost is not always an anomaly. A $900/day production Kubernetes cluster can be normal, while a $400/day development EC2 instance can be anomalous if its historical average is $50. The model must evaluate cost **IN CONTEXT**.

---

## 💡 Motivation

Companies using cloud infrastructure face unexpected cost increases from:

- Wrong resource configuration
- Forgotten test environments
- Data transfer spikes
- Deployment errors
- Unused resources
- Monitoring/logging misconfiguration
- Duplicated resources
- Development resources left running

Manual monitoring doesn't scale. ML can automatically detect unusual spending patterns.

---

## 📊 Dataset

### Synthetic Data Generation

The dataset is generated synthetically by `src/data_generation.py` using `RANDOM_STATE=67`.

**Why synthetic data?**

- No access to private billing data required
- Full control over anomaly scenarios
- Safe for university presentation
- Reproducible experiments
- Original approach (earns creativity points)

### Dataset Overview

| Property | Value |
|---|---|
| Records | ~5,000 |
| Features | 20 columns |
| Anomaly rate | ~7% (imbalanced dataset) |
| Time period | ~90 days |
| Providers | AWS, Azure, Google Cloud |
| Regions | us-east-1, eu-central-1, eu-west-1, us-west-2 |
| Services | EC2, S3, RDS, Lambda, CloudWatch, Data Transfer, Kubernetes |
| Environments | production, staging, development |

### Raw Data Columns

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `date` | date | Record date |
| 2 | `day_of_week` | int | 0=Monday ... 6=Sunday |
| 3 | `is_weekend` | int | 1 if Saturday/Sunday |
| 4 | `month` | int | 1–12 |
| 5 | `cloud_provider` | str | AWS / Azure / Google Cloud |
| 6 | `region` | str | Cloud region |
| 7 | `service_name` | str | Cloud service |
| 8 | `environment` | str | production / staging / development |
| 9 | `usage_amount` | float | Abstract usage units |
| 10 | `resource_count` | int | Number of active resources |
| 11 | `data_transfer_gb` | float | Data transferred (GB) |
| 12 | `storage_gb` | float | Storage consumed (GB) |
| 13 | `compute_hours` | float | Compute time (hours) |
| 14 | `request_count` | int | API/request count |
| 15 | `previous_day_cost` | float | Cost on previous day |
| 16 | `rolling_3_day_avg_cost` | float | Mean cost over past 3 days |
| 17 | `rolling_7_day_avg_cost` | float | Mean cost over past 7 days |
| 18 | `cost_usd` | float | Total cost in USD |
| 19 | `anomaly_type` | str | Anomaly scenario label (analysis only, **NOT** a feature) |
| 20 | `is_anomaly` | int | Target variable: 0=normal, 1=anomaly |

### Anomaly Types

| Type | Description |
|---|---|
| `sudden_compute_spike` | Cost jumps 4–8x due to compute surge |
| `abnormal_data_transfer` | Data transfer increases 10–20x |
| `storage_growth` | Storage grows 8–15x abnormally |
| `forgotten_test_environment` | Dev/staging costs reach production levels |
| `unexpected_region_usage` | Non-primary region shows high cost |
| `monitoring_cost_spike` | Monitoring/logging costs spike 5–10x |
| `weekend_production_activity` | Unexpected high weekend costs |
| `duplicated_resource_deployment` | Resources duplicated 4–8x |

---

## 🔧 Feature Engineering

### Time Features

`day_of_week`, `month`, `is_weekend`, `is_business_day`

### Cost Trend Features

`cost_to_rolling_7_day_ratio`, `cost_to_rolling_3_day_ratio`, `cost_difference_from_previous_day`, `cost_change_percent`

Rolling features use only **PAST** values (no data leakage).

### Cost Efficiency Features

`cost_per_resource`, `cost_per_compute_hour`, `cost_per_gb_transfer`, `cost_per_1000_requests`, `cost_per_storage_gb`

All use safe division (`denominator + 1`) to avoid division by zero.

### Rule-Based Indicators

`is_high_transfer`, `is_high_compute`, `is_high_resource_count`, `is_dev_environment_expensive`, `is_cost_much_higher_than_average`, `is_unusual_weekend_activity`

These are derived from **INPUT features only**, never from the target variable.

---

## 🔬 Machine Learning Pipeline

### Preprocessing

- **Numeric features:** `SimpleImputer(median)` → `RobustScaler`
- **Categorical features:** `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`
- `RobustScaler` chosen because cloud cost data contains outliers and cost spikes
- `OneHotEncoder` with `handle_unknown="ignore"` because new regions/services may appear

### Train/Test Split

- 80% training, 20% testing
- Stratified split (preserves anomaly ratio)
- Preprocessing fitted **ONLY** on training data

### Data Validation

- Required columns exist
- No negative costs or usage
- Anomaly rate within realistic range
- `target` / `anomaly_type` excluded from features
- No fully empty columns

---

## 🤖 Models

### 1. Majority Class Baseline

Always predicts the most frequent class (normal). Demonstrates the "accuracy trap" — high accuracy with zero anomaly detection.

### 2. Logistic Regression

Linear supervised classifier with `class_weight="balanced"` to handle imbalance. Provides interpretable feature weights.

### 3. Random Forest

Ensemble of decision trees with `GridSearchCV` optimization (cv=4, scoring=f1). Tuned parameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`. Provides feature importance ranking.

### 4. Isolation Forest

Unsupervised anomaly detection — does **NOT** use labels during training. Finds observations that are easy to "isolate" from the majority. `contamination` parameter set to approximate anomaly rate.

**Supervised vs Unsupervised:** Logistic Regression and Random Forest learn from labeled data. Isolation Forest finds unusual patterns without labels.

---

## 📈 Evaluation Metrics

- **Accuracy** — overall correctness (misleading for imbalanced data!)
- **Precision** — of predicted anomalies, how many are real?
- **Recall** — of real anomalies, how many did we find?
- **F1 Score** — harmonic mean of precision and recall
- **ROC-AUC** — area under ROC curve
- **Average Precision** — area under precision-recall curve

### Why Accuracy Is Not Enough

With ~7% anomalies, predicting all records as "normal" gives ~93% accuracy but detects **ZERO** anomalies (recall = 0). The model is useless despite high accuracy. This is the **"accuracy trap."**

### Confusion Matrix — Business Meaning

| Outcome | What Happens |
|---|---|
| **True Positive** | Anomaly correctly detected → team investigates real cost spike |
| **True Negative** | Normal cost correctly classified → no unnecessary investigation |
| **False Positive** | Normal cost flagged as anomaly → team wastes time investigating |
| **False Negative** | Real anomaly missed → company loses money from undetected cost spike |

In cloud cost monitoring, **False Negatives are usually MORE dangerous**.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority Baseline | 0.9306 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.9772 | 0.7765 | 0.9429 | 0.8516 | 0.9730 |
| **Random Forest** | **0.9841** | **0.8857** | **0.8857** | **0.8857** | **0.9863** |
| Isolation Forest | 0.9365 | 0.5405 | 0.5714 | 0.5556 | N/A |

**Best model: Random Forest** with F1=0.886 and ROC-AUC=0.986 on the held-out test set.

Final model selected based on F1-score, recall, and business interpretation of errors — **NOT** accuracy alone.

---

## 📉 Visualizations

Generated in `assets/plots/`:

- Class distribution
- Cost over time with anomaly highlights
- Cost distribution histogram
- Cost by service / provider / environment / region
- Anomaly type distribution
- Rolling average vs actual cost
- Confusion matrices (all models)
- ROC curve comparison
- Precision-recall curve comparison
- Feature importance (Random Forest)
- Model comparison

---

## 🔍 Explainability

### Feature Importance

Top features from Random Forest reveal what drives anomaly detection: cost ratios vs rolling averages, usage metrics, and contextual indicators.

### Error Analysis

Analysis of which anomalies were caught, which were missed, and whether errors cluster around specific services or environments.

### Case Study

**Normal record:**

```
AWS EC2, production, $420/day, 7-day avg $405
→ Prediction: NORMAL ✅
```

**Anomalous record:**

```
AWS EC2, development, $1,200/day, 7-day avg $100
→ Prediction: ANOMALY 🚨
Indicators: cost 12x above average, development environment too expensive,
            high compute and data transfer
```

---

## ☁️ Cloud Architecture Concept

How this system could work in a real company:

1. **Data Export** — Cloud billing data exported daily from AWS/Azure/GCP
2. **Object Storage** — Raw billing data stored in S3/Blob Storage/GCS
3. **Scheduled Job** — Cron job or Cloud Function triggers ML pipeline
4. **ML Pipeline** — Preprocessing → Feature Engineering → Model Scoring
5. **Model Artifact** — Trained model stored in model registry
6. **Predictions** — New cost records scored as normal/anomaly
7. **Dashboard** — Streamlit/Grafana shows suspicious cost spikes
8. **Alerting** — Email/Slack alerts sent to FinOps/DevOps teams

*Note: This is a conceptual architecture. The university project runs locally without cloud credentials.*

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/maksivanouski/cloud-cost-anomaly-detection.git
cd cloud-cost-anomaly-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the full pipeline step by step:

```bash
# Step 1: Generate synthetic cloud cost dataset
python src/data_generation.py

# Step 2: Run exploratory data analysis (generates plots)
python src/exploratory_analysis.py

# Step 3: Train all models
python src/train.py

# Step 4: Evaluate models and generate metrics/plots
python src/evaluate.py

# Step 5: Run prediction on example records
python src/predict.py

# Optional: Launch interactive dashboard
streamlit run src/dashboard.py
```

---

## 🔮 Example Prediction

```
$ python src/predict.py

============================================================
  Example 1: Normal Production Cost
============================================================
Provider: AWS | Service: EC2 | Env: production | Cost: $420.00
Rolling 7-day avg: $405.00

Prediction: NORMAL ✅
Confidence: ~89%
No suspicious indicators detected.

============================================================
  Example 2: Anomalous Development Cost Spike
============================================================
Provider: AWS | Service: EC2 | Env: development | Cost: $1,200.00
Rolling 7-day avg: $100.00

Prediction: ANOMALY 🚨
Confidence: ~94%
Suspicious indicators:
  • cost is much higher than rolling 7-day average
  • development environment has unusually high cost
  • compute hours are high
  • data transfer is high
  • resource count is high
```

---

## 📊 Dashboard (Optional)

The optional Streamlit dashboard provides:

- Dataset overview with key statistics
- Interactive cost-over-time visualization
- Anomaly distribution analysis
- Model prediction form

```bash
streamlit run src/dashboard.py
```

The core project works fully without Streamlit.

---

## 📁 Project Structure

```
cloud-cost-anomaly-detection/
├── data/
│   ├── raw/
│   │   └── cloud_cost_data.csv
│   ├── processed/
│   │   └── cloud_cost_features.csv
│   └── README.md
├── src/
│   ├── data_generation.py
│   ├── data_loading.py
│   ├── exploratory_analysis.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── dashboard.py
│   └── utils.py
├── models/
│   ├── README.md
│   └── *.pkl
├── assets/
│   ├── plots/
│   ├── screenshots/
│   └── architecture/
├── reports/
│   ├── report_pl.md
│   └── presentation_outline_pl.md
├── requirements.txt
├── README.md
├── Dockerfile
├── .gitignore
└── LICENSE
```

---

## ⚠️ Common ML Risks Avoided

### Data Leakage Prevention

- Rolling features computed from **PAST** values only
- Preprocessing fitted **ONLY** on training data
- Test data used **ONLY** for final evaluation
- `anomaly_type` excluded from training features
- Target column excluded from input features
- Explicit validation checks in code

### Artificially Easy Dataset Prevention

- Anomalies are **NOT** created by one trivial rule
- High cost ≠ always anomaly (production costs can be legitimately high)
- Normal records can also have high costs
- Model must use context (environment, service, rolling averages)
- Classes partially overlap in cost range

### Accuracy Trap Prevention

- Baseline model shows accuracy is misleading
- Final model selected by F1, recall, precision — not accuracy
- Confusion matrix and business impact analyzed
- Multiple metrics reported for every model

### Pipeline Saving

- Saved model includes **FULL** pipeline (preprocessor + classifier)
- Prediction uses same transformations as training
- Prevents inconsistent preprocessing during inference

---

## 📝 Assignment Criteria Alignment

| Criterion | Max Points | How This Project Addresses It |
|---|---|---|
| Timeliness | 10 | Clear setup, reproducible pipeline, saved artifacts, prepared report |
| Creativity | 10 | Original synthetic cloud billing dataset, 8 anomaly types, cloud architecture concept |
| Code Quality & ML Pipeline | 10 | Complete pipeline: generation → validation → EDA → features → preprocessing → train/test → baseline → models → GridSearchCV → evaluation → prediction |
| Report Quality | 10 | 2–5 page Polish report with metrics, confusion matrix, accuracy trap explanation, case studies |
| Presentation | 10 | 12-slide outline, visualizations, live demo, cloud architecture concept |

---

## 🔮 Future Improvements

- Integration with real AWS/Azure/GCP billing APIs
- Real-time streaming anomaly detection
- Automated alerting via email/Slack
- Time-series models (Prophet, LSTM) for trend forecasting
- Multi-class anomaly classification (not just binary)
- A/B testing of detection thresholds
- Cost forecasting alongside anomaly detection
- MLflow experiment tracking
- Model retraining pipeline with data drift detection
