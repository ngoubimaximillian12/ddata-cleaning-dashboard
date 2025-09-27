# 🧹 Intelligent Data Cleaning & Evaluation Dashboard

A comprehensive **Streamlit application** that merges **traditional data cleaning methods** with **advanced ML techniques** (autoencoders for anomaly detection, adaptive method recommendations, and real‑time model evaluation).

---

## ✨ Features

### Core Functionality

* **Multi‑Method Cleaning**: Traditional, ML‑based, and autoencoder approaches
* **Intelligent Recommendations**: Adaptive suggestions based on dataset characteristics
* **Real‑Time Model Evaluation**: Built‑in ML model testing with performance metrics
* **Interactive Dashboard**: User‑friendly Streamlit interface
* **Automated Reporting**: PDF export of cleaning summaries
* **Learning System**: Continuously improves recommendations from usage patterns

### Advanced Capabilities

* Autoencoder‑based anomaly detection
* Regex text pattern cleaning
* Multi‑domain support (finance, healthcare, education, general)
* Adaptive contamination thresholds
* Comprehensive performance benchmarking

---

## ⚙️ Installation

### Prerequisites

```bash
Python 3.8+
pip or conda package manager
```

### Required Dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn dask scikit-learn tensorflow fpdf2
```

### Quick Setup

```bash
# Clone the repository
git clone <repository-url>
cd data-cleaning-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 🚀 Usage

### Basic Workflow

1. **Upload Dataset** (CSV, any domain)
2. **Configure Settings**

   * Target column for evaluation
   * Cleaning method: Auto / Traditional / ML‑Based / Autoencoder
   * Sidebar parameter adjustments
3. **Review Results**

   * Cleaning metrics + model performance
   * Before/after dataset comparison
   * PDF summary download

### Cleaning Methods

* **Traditional**: Mean imputation + duplicate removal
* **ML‑Based**: KNN imputation + Isolation Forest
* **Autoencoder**: Neural anomaly detection
* **Auto**: Adaptive system recommendation

---

## 🏗️ Architecture

### Core Components

1. **Data Cleaning Engine**

```python
def autoencoder_cleaning(df, threshold=0.01):
    """
    Autoencoder‑based anomaly detection for data cleaning
    """
```

2. **Adaptive Recommendation System**

```python
def suggest_cleaning_method(df):
    """
    Recommends optimal cleaning method based on dataset characteristics.
    """
```

3. **Real‑Time Evaluation**

```python
def evaluate_models(df, target_col, model_choice):
    """
    ML evaluation with metrics + predictions.
    """
```

### Autoencoder Design

* Input → Dense(16, relu, L1 reg) → Dense(n_features, linear)
* Training: 20 epochs, batch=32, Adam, MSE loss
* Anomaly detection: remove rows with high reconstruction error

---

## 📊 Performance Metrics

### Cleaning Effectiveness

* Missing value reduction
* Outlier detection rate
* Data retention
* Processing speed

### Model Performance

* Classification: Accuracy, F1, Confusion Matrix
* Regression: RMSE, MAE, R²
* Cross‑validation & residual analysis

### Composite Score

```
Cleaning Score = 0.4 × Missing_Reduction + 0.3 × Outlier_Reduction + 0.3 × Model_Performance
```

---

## ⚡ Configuration Examples

### Financial Data

```python
cleaning_config = {
    "method": "autoencoder",
    "contamination": 0.02,
    "regex_rules": [{"column": "symbol", "pattern": r"[^A-Z]", "replacement": ""}]
}
```

### Healthcare Data

```python
cleaning_config = {
    "method": "ml_based",
    "knn_neighbors": 5,
    "apply_outliers": False,
    "regex_rules": [{"column": "patient_id", "pattern": r"[^0-9]", "replacement": ""}]
}
```

### Educational Data

```python
cleaning_config = {
    "method": "traditional",
    "regex_rules": [{"column": "grade", "pattern": r"[^A-F\+\-]", "replacement": ""}]
}
```

---

## 📦 Technical Specs

### System Requirements

* RAM: 4GB min (8GB recommended)
* CPU: Multi‑core
* Storage: 1GB free
* Network: Required for setup

### Performance Benchmarks

* Small (<10K rows): 1–5s
* Medium (10K–100K): 5–30s
* Large (>100K): 30–300s + 10–60s for autoencoder

### Scalability

* **Dask** integration for large datasets
* Incremental learning & caching support

---

## 🔧 Troubleshooting

### Common Issues

* **Memory Errors** → Use Dask
* **CUDA/GPU Issues** → Disable GPU via env vars
* **Low Accuracy** → Adjust contamination or cleaning method
* **Poor Recommendations** → Reset `learning_log.csv`

Error handling covers invalid formats, missing targets, small datasets, and corrupted files.

---

## 🤝 Contributing

### Development Setup

```bash
git clone <repository-url>
cd data-cleaning-dashboard
pip install -r requirements-dev.txt
```

### Testing

```bash
python -m pytest tests/
python -m pytest tests/integration/
python benchmarks/performance_test.py
```

### Code Style

* **PEP 8** formatting
* **Type hints** required
* **Docstrings** mandatory
* **Explicit exception handling**

---

## 📄 License

Licensed under the **MIT License** — see `LICENSE`.

---

## 👨‍💻 Author

**Ngoubi Maximillian Diamgha**
GitHub: [@ngoubimaximillian12](https://github.com/ngoubimaximillian12)
Email: [ngoubimaximilliandiangha@gmail.com](mailto:ngoubimaximilliandiangha@gmail.com)
LinkedIn: [Diangha Ngoubi](https://www.linkedin.com/in/diangha-ngoubi-42a49b281/)

---

## 🙏 Acknowledgments

* TensorFlow/Keras
* Scikit‑learn
* Streamlit
* Pandas / NumPy
* Research community in anomaly detection

---

## 📋 Changelog

### Version 1.0.0 — Current

* Initial release
* Autoencoder anomaly detection
* Adaptive recommendation system
* Real‑time evaluation
* PDF export
* Full documentation

---

## ⚡ Quick Start Example

```python
import streamlit as st
import pandas as pd

# Load dataset
df = pd.read_csv("your_data.csv")

# Run the Streamlit app
streamlit run app.py
```

Steps:

1. Upload CSV
2. Select target column
3. Choose cleaning method (or Auto)
4. Review results + download report

**Expected Processing Time:** 30s – 5min (depends on dataset + method).
**Supported Data Types:** numerical, categorical, text, datetime, mixed.
**Use Cases:** preprocessing, EDA, model prep, automated data quality assessment.
