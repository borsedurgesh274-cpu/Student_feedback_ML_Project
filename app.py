# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import io
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

st.set_page_config(page_title="Model Demo (Student_model)", layout="wide")
st.title("Streamlit → Load `Student_model.pkl` & `employee_clean_data.csv`")

#
# Default paths (where your uploaded files are placed in this environment)
#
DEFAULT_MODEL_PATH = "/mnt/data/Student_model.pkl"
DEFAULT_CSV_PATH = "/mnt/data/employee_clean_data.csv"

#
# Helpers
#
@st.cache_data
def load_model(path_bytes_or_str):
    """Load model from a path or file-like bytes. Try joblib then pickle."""
    try:
        if isinstance(path_bytes_or_str, (bytes, io.BytesIO)):
            # joblib accepts file-like; but joblib.load expects file-like with seek
            return joblib.load(io.BytesIO(path_bytes_or_str))
        else:
            return joblib.load(path_bytes_or_str)
    except Exception:
        try:
            if isinstance(path_bytes_or_str, (bytes, io.BytesIO)):
                return pickle.load(io.BytesIO(path_bytes_or_str))
            else:
                with open(path_bytes_or_str, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

@st.cache_data
def load_csv(path_bytes_or_str):
    try:
        if isinstance(path_bytes_or_str, (bytes, io.BytesIO)):
            return pd.read_csv(io.BytesIO(path_bytes_or_str))
        else:
            return pd.read_csv(path_bytes_or_str)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")

def infer_features(model, df):
    # try common attributes, else fallback to df columns except last (heuristic)
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
    except Exception:
        pass
    if df is not None:
        cols = list(df.columns)
        if len(cols) >= 2:
            return cols[:-1]  # guess last is target
        return cols
    return []

def safe_predict(model, X):
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
        except Exception:
            probs = None
    return preds, probs

#
# Sidebar: file inputs
#
st.sidebar.header("Files / Options")
uploaded_model = st.sidebar.file_uploader("Upload model (.pkl/.joblib) — otherwise app loads default", type=["pkl","joblib"])
uploaded_csv = st.sidebar.file_uploader("Upload CSV — otherwise app loads default", type=["csv"])
run_bulk_pred = st.sidebar.checkbox("Enable bulk prediction (use CSV + model)", value=True)
show_raw = st.sidebar.checkbox("Show raw CSV", value=False)

#
# Load model
#
model = None
model_err = None
try:
    if uploaded_model is not None:
        model = load_model(uploaded_model.read())
        st.sidebar.success("Model uploaded and loaded.")
    else:
        if Path(DEFAULT_MODEL_PATH).exists():
            model = load_model(DEFAULT_MODEL_PATH)
            st.sidebar.success(f"Loaded model from {DEFAULT_MODEL_PATH}")
        else:
            st.sidebar.warning("No model uploaded and default model not found.")
except Exception as e:
    model_err = str(e)
    st.sidebar.error(f"Model load error: {model_err}")

#
# Load CSV
#
df = None
csv_err = None
try:
    if uploaded_csv is not None:
        df = load_csv(uploaded_csv.read())
        st.sidebar.success("CSV uploaded and loaded.")
    else:
        if Path(DEFAULT_CSV_PATH).exists():
            df = load_csv(DEFAULT_CSV_PATH)
            st.sidebar.success(f"Loaded CSV from {DEFAULT_CSV_PATH}")
        else:
            st.sidebar.warning("No CSV uploaded and default CSV not found.")
except Exception as e:
    csv_err = str(e)
    st.sidebar.error(f"CSV load error: {csv_err}")

#
# Show model & data status
#
st.markdown("### Status")
if model is not None:
    st.write("**Model type:**", type(model).__name__)
else:
    st.error("Model not loaded.")

if df is not None:
    st.write("**CSV shape:**", df.shape)
    if show_raw:
        st.dataframe(df.head(200))
else:
    st.info("CSV not loaded.")

st.markdown("---")
#
# Feature detection and single-row prediction form
#
st.header("Single-row prediction")
features = infer_features(model, df)
if features:
    st.write(f"Detected features ({len(features)}): {features}")
else:
    st.info("Could not detect features automatically. You can type feature names below.")

# Build form
input_values = {}
if features:
    with st.form("single_predict"):
        cols = st.columns(2)
        for i, f in enumerate(features):
            col = cols[i % 2]
            if df is not None and f in df.columns:
                if pd.api.types.is_numeric_dtype(df[f].dtype):
                    val = col.number_input(f, value=float(df[f].dropna().iloc[0]) if not df[f].dropna().empty else 0.0)
                    input_values[f] = val
                else:
                    uniques = df[f].dropna().unique().tolist()
                    if 1 < len(uniques) <= 50:
                        input_values[f] = col.selectbox(f, options=map(str, uniques))
                    else:
                        input_values[f] = col.text_input(f, value=str(uniques[0]) if uniques else "")
            else:
                input_values[f] = col.text_input(f, value
