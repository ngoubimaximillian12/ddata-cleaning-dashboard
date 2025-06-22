# 🧹 Data Cleaning Dashboard

An intelligent Streamlit-based web app for automated and ML-enhanced data cleaning, evaluation, and recommendation. Supports traditional methods, ML-based strategies, and deep learning (Autoencoders), with learning-based recommendations and PDF summaries.

---

## 🚀 Features

- 📂 Upload and preview CSV files
- 🧼 Choose from:
  - Traditional Cleaning
  - ML-Based Cleaning
  - Autoencoder Cleaning
  - Auto mode (with recommender)
- 🔍 Regex-based text column cleaning
- 🧠 Cleaning method recommender (trained on logs)
- 📈 Model evaluation (classification & regression)
- 🧮 Cleaning score with metrics
- 🧾 PDF summary export
- 🔁 Learns from each run to improve recommendations

---

## 📦 Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
If deploying on Streamlit Cloud, make sure .streamlit/config.toml exists:

toml
Copy
Edit
[general]
pythonVersion = "3.11"
🧪 Running Locally
bash
Copy
Edit
streamlit run full_data_cleaning_dashboard_final.py
🧠 Learning & Recommender
Each cleaning job is logged to learning_log.csv with features like:

Number of rows/columns

Missing % and duplicates

Numeric/categorical ratio

Chosen method and score

A simple Random Forest classifier is trained every 10 runs to improve future cleaning recommendations.

📁 Folder Structure
bash
Copy
Edit
.
├── full_data_cleaning_dashboard_final.py   # Streamlit app and logic
├── learning_log.csv                        # Cleaning logs
├── recommender.pkl                         # Trained recommendation model
├── requirements.txt                        # Dependencies
└── .streamlit/
    └── config.toml                         # Python version for Streamlit Cloud
🧠 Tech Stack
Streamlit

Pandas, NumPy, Scikit-learn, Dask

TensorFlow (Autoencoders)

fpdf (for PDF reports)

☁️ Deploying to Streamlit Cloud
Push your code to GitHub

Go to Streamlit Cloud

Click "New App"

Select your repo (e.g., ngoubimaximillian12/ddata-cleaning-dashboard)

Use full_data_cleaning_dashboard_final.py as the entry point

Ensure your repo has .streamlit/config.toml and correct requirements.txt

🛑 Known Limitations
Autoencoder-based cleaning may be slow on large datasets

TensorFlow requires compatible Python (<3.13)

Currently supports CSV input only

🤝 Contributing
Contributions are welcome! Fork the repo, make your changes, and submit a pull request.

📄 License
MIT License © 2025 Ngoubi Maximillian

yaml
Copy
Edit

---

Let me know if you’d like this saved into a file and pushed to your GitHub automatically.







