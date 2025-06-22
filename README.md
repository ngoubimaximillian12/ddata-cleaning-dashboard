from pathlib import Path

readme_content = """# 🧹 Data Cleaning Dashboard

An intelligent **Streamlit-based** web app for automated and ML-enhanced data cleaning, evaluation, and recommendation. Supports traditional, ML-based, and deep learning (Autoencoder) cleaning, with a learning system that improves suggestions over time and generates PDF summaries.

---

## 🚀 Features

- 📂 Upload and preview CSV files
- 🧼 Choose from cleaning strategies:
  - Traditional Cleaning
  - ML-Based Cleaning (KNN + Isolation Forest)
  - Autoencoder Cleaning (deep learning)
  - Auto Mode (uses recommender system)
- 🔍 Regex-based text cleaning for specific columns
- 🧠 Cleaning method recommender trained on usage logs
- 📈 Evaluate models (classification or regression)
- 🧮 Calculates a composite **cleaning score**
- 🧾 Generate PDF summary of corrections made
- 🔁 Learns from each run and improves over time

---

## 📦 Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
