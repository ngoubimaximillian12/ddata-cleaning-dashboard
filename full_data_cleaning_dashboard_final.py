import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Full Streamlit App with Cleaning Logic, Autoencoder, Recommender, and UI

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd
import pickle
import time
import csv
import re
from io import BytesIO, StringIO
from datetime import datetime
from fpdf import FPDF
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# ---------------------- Autoencoder-Based Cleaning ----------------------
def autoencoder_cleaning(df, threshold=0.01):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    X = df[numeric_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation="relu", activity_regularizer=regularizers.l1(10e-5))(input_layer)
    decoded = Dense(input_dim, activation="linear")(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(), loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=20, batch_size=32, shuffle=True, verbose=0)
    reconstructions = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    mask = mse < np.quantile(mse, 1 - threshold)
    return df.loc[X.index[mask]]

# ---------------------- PDF Export Summary ----------------------
def export_pdf_summary(summary_dict, filename="cleaning_summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Data Cleaning Summary", ln=True, align="C")
    for k, v in summary_dict.items():
        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)
    pdf.output(filename)
    return filename

# ---------------------- Utility Functions ----------------------
DOMAIN_NOTE = "This dashboard supports datasets from finance, health, education and more."

def summarize_corrections(original_df, cleaned_df):
    summary = {}
    summary["Initial Shape"] = original_df.shape
    summary["Final Shape"] = cleaned_df.shape
    summary["Rows Removed"] = original_df.shape[0] - cleaned_df.shape[0]
    summary["Duplicates Removed"] = original_df.duplicated().sum()
    orig_missing = original_df.isnull().sum().sum()
    clean_missing = cleaned_df.isnull().sum().sum()
    summary["Missing Values Filled"] = int(orig_missing - clean_missing)
    return summary

def extract_dataset_features(df):
    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing_pct": df.isnull().sum().sum() / (df.size + 1e-6),
        "duplicate_pct": df.duplicated().sum() / (len(df) + 1e-6),
        "num_cols": len(df.select_dtypes(include=np.number).columns),
        "cat_cols": len(df.select_dtypes(include='object').columns),
    }

def train_recommender(log_df):
    if len(log_df) < 5:
        return None, None
    feats = log_df[["n_rows", "n_cols", "missing_pct", "duplicate_pct", "num_cols", "cat_cols"]]
    target = log_df["Recommended Method"]
    model = RandomForestClassifier()
    model.fit(feats, target)
    with open("recommender.pkl", "wb") as f:
        pickle.dump(model, f)
    return model, feats.columns.tolist()

def load_recommender():
    try:
        with open("recommender.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

def suggest_cleaning_method(df):
    model = load_recommender()
    if not model:
        return "Auto"
    features = extract_dataset_features(df)
    X = pd.DataFrame([features])
    return model.predict(X)[0]

def log_learning_run(df, method, score, retrain_every=10):
    features = extract_dataset_features(df)
    features["Recommended Method"] = method
    features["Cleaning Score"] = score
    try:
        existing = pd.read_csv("learning_log.csv")
        log_df = pd.concat([existing, pd.DataFrame([features])], ignore_index=True)
    except FileNotFoundError:
        log_df = pd.DataFrame([features])
    log_df.to_csv("learning_log.csv", index=False)
    if len(log_df) % retrain_every == 0:
        train_recommender(log_df)

# ---------------------- Cleaning Methods ----------------------
def regex_cleaning(df, regex_rules):
    df = df.copy()
    for rule in regex_rules:
        col = rule.get("column")
        pattern = rule.get("pattern")
        repl = rule.get("replacement", "")
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: re.sub(pattern, repl, x))
    return df

def traditional_cleaning(df, regex_rules=[]):
    df = regex_cleaning(df, regex_rules)
    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    return df

def ml_cleaning(df, contamination=0.05, knn_n=3, apply_outlier=True, regex_rules=[]):
    df = regex_cleaning(df, regex_rules)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = KNNImputer(n_neighbors=knn_n).fit_transform(df[num_cols])
    if apply_outlier:
        preds = IsolationForest(contamination=contamination).fit_predict(df[num_cols])
        df = df[preds == 1]
    return df

def encode_categoricals(df, strategy="label"):
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

# ---------------------- Evaluation ----------------------
def calculate_cleaning_score(original_df, cleaned_df, model_metric_value, is_classification):
    missing_before = original_df.isnull().sum().sum()
    missing_after = cleaned_df.isnull().sum().sum()
    missing_reduction = 1 - (missing_after / (missing_before + 1e-6))
    size_before = len(original_df)
    size_after = len(cleaned_df)
    outlier_reduction = 1 - (size_before - size_after) / (size_before + 1e-6)
    score = 0.4 * missing_reduction + 0.3 * outlier_reduction + 0.3 * model_metric_value
    return round(score, 3)

def evaluate_models(df, target_col, model_choice):
    start_time = time.time()
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    classification = pd.api.types.is_categorical_dtype(y) or y.nunique() < 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if classification:
        model = RandomForestClassifier() if model_choice == "Random Forest" else (
            KNeighborsClassifier() if model_choice == "KNN" else LogisticRegression(max_iter=200))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metric = accuracy_score(y_test, pred)
        results = {"Accuracy": metric, "F1": f1_score(y_test, pred, average='weighted')}
    else:
        model = RandomForestRegressor() if model_choice == "Random Forest" else (
            KNeighborsRegressor() if model_choice == "KNN" else LinearRegression())
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metric = r2_score(y_test, pred)
        results = {"RMSE": mean_squared_error(y_test, pred, squared=False), "R2": metric}
    results["Execution Time (s)"] = round(time.time() - start_time, 2)
    return results, classification, metric

# ---------------------- Streamlit UI ----------------------
st.set_page_config("Data Cleaning Dashboard", layout="wide")
st.title("🧹 Intelligent Data Cleaning & Evaluation App")

st.markdown(DOMAIN_NOTE)

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Raw Dataset Preview")
    st.dataframe(df.head())

    target_col = st.selectbox("Select your target column", options=df.columns)
    model_type = st.selectbox("Choose ML model", ["Random Forest", "KNN", "Logistic/Linear"])

    st.sidebar.header("⚙️ Cleaning Config")
    cleaning_method = st.sidebar.selectbox("Method", ["Auto", "Traditional", "ML-Based", "Autoencoder"])
    contam = st.sidebar.slider("Isolation Forest contamination", 0.01, 0.2, 0.05)
    knn_n = st.sidebar.slider("KNN Imputer Neighbors", 1, 10, 3)
    apply_outliers = st.sidebar.checkbox("Apply Outlier Detection", value=True)
    regex_json = st.sidebar.text_area("Regex Cleaning Rules (JSON Array)", "[]")
    regex_rules = eval(regex_json) if regex_json else []

    suggested = suggest_cleaning_method(df)
    st.sidebar.markdown(f"💡 Suggested Method: **{suggested}**")

    df_encoded = encode_categoricals(df.copy())

    if cleaning_method == "Traditional" or (cleaning_method == "Auto" and suggested == "Traditional"):
        df_cleaned = traditional_cleaning(df_encoded.copy(), regex_rules)
    elif cleaning_method == "ML-Based" or (cleaning_method == "Auto" and suggested == "ML-Based"):
        df_cleaned = ml_cleaning(df_encoded.copy(), contam, knn_n, apply_outliers, regex_rules)
    else:
        df_cleaned = autoencoder_cleaning(df_encoded.copy())

    eval_results, is_classification, eval_metric = evaluate_models(df_cleaned, target_col, model_type)
    cleaning_score = calculate_cleaning_score(df_encoded, df_cleaned, eval_metric, is_classification)
    corrections = summarize_corrections(df_encoded, df_cleaned)

    st.subheader("📊 Evaluation Metrics")
    st.json(eval_results)
    st.metric("Cleaning Score", cleaning_score)

    st.subheader("🧾 Cleaning Summary")
    st.json(corrections)

    log_learning_run(df_encoded, cleaning_method, cleaning_score)

    if st.button("📥 Export PDF Summary"):
        filename = export_pdf_summary(corrections)
        with open(filename, "rb") as f:
            st.download_button("Download PDF", f, file_name=filename)

    st.success("✅ Cleaning complete. Explore and export results above.")
