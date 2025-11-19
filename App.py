import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from pathlib import Path

st.set_page_config(page_title="Model Predictor", layout="wide")
st.title("📈 Streamlit Predictor — Student_model.pkl")

# Default paths
DEFAULT_MODEL = Path("Student_model.pkl")
DEFAULT_CSV = Path("employee_clean_data.csv")

# Sidebar - allow uploads or use defaults
st.sidebar.header("Files")
model_u = st.sidebar.file_uploader("Upload model (.pkl)", type=["pkl"])
csv_u = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_defaults = False
if model_u is None and csv_u is None:
    use_defaults = True
    st.sidebar.info(f"Using defaults if found:\n• {DEFAULT_MODEL}\n• {DEFAULT_CSV}")

@st.cache_data
def read_csv_from_bytes(b: bytes):
    return pd.read_csv(io.BytesIO(b))

@st.cache_resource
def load_model_from_bytes(b: bytes):
    return pickle.load(io.BytesIO(b))

@st.cache_resource
def load_model_from_path(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

# Load CSV (for sample values / columns)
df = None
if csv_u is not None:
    try:
        df = read_csv_from_bytes(csv_u.read())
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    if DEFAULT_CSV.exists():
        try:
            df = pd.read_csv(DEFAULT_CSV)
        except Exception as e:
            st.warning(f"Could not read default CSV: {e}")

# Load model
model = None
if model_u is not None:
    try:
        model = load_model_from_bytes(model_u.read())
    except Exception as e:
        st.error(f"Failed to load uploaded model: {e}")
        st.stop()
else:
    if DEFAULT_MODEL.exists():
        try:
            model = load_model_from_path(DEFAULT_MODEL)
        except Exception as e:
            st.error(f"Failed to load default model: {e}")
            st.stop()
    else:
        st.warning("No model uploaded and no default model found. Upload a .pkl model in the sidebar.")
        st.stop()

# Helper: infer feature names
def infer_features(m, sample_df):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)

    try:
        if hasattr(m, "named_steps"):
            last = list(m.named_steps.items())[-1][1]
            if hasattr(last, "feature_names_in_"):
                return list(last.feature_names_in_)
        if hasattr(m, "steps"):
            last = m.steps[-1][1]
            if hasattr(last, "feature_names_in_"):
                return list(last.feature_names_in_)
    except Exception:
        pass

    if hasattr(m, "feature_names"):
        return list(getattr(m, "feature_names"))

    if sample_df is not None:
        cols = list(sample_df.columns)
        for t in ["target", "label", "salary", "y", "outcome"]:
            cols = [c for c in cols if c.lower() != t]
        return cols

    return []

feature_names = infer_features(model, df)
if not feature_names:
    st.error("Could not infer feature names. Upload a CSV with example rows or provide a model with feature_names_in_.")
    st.stop()

st.sidebar.markdown(f"Detected **{len(feature_names)}** features")

# Input form
st.subheader("Provide input values")
with st.form("predict"):
    inputs = {}
    for col in feature_names:
        if df is not None and col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                min_v = float(series.min()) if not np.isnan(series.min()) else 0.0
                max_v = float(series.max()) if not np.isnan(series.max()) else min_v + 100.0
                default_v = float(series.median()) if not np.isnan(series.median()) else (min_v + max_v) / 2
                inputs[col] = st.number_input(
                    col,
                    value=default_v,
                    min_value=min_v,
                    max_value=max_v,
                    step=(max_v - min_v) / 100 if max_v != min_v else 1.0,
                )
            else:
                opts = sorted(series.dropna().unique().tolist())
                if len(opts) <= 50:
                    inputs[col] = st.selectbox(col, options=[""] + opts)
                else:
                    inputs[col] = st.text_input(col, value="")
        else:
            inputs[col] = st.text_input(col, value="")

    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([inputs], columns=feature_names)
    st.write("### Input preview")
    st.dataframe(X)

    for c in X.columns:
        if df is not None and c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        else:
            X[c] = pd.to_numeric(X[c], errors="ignore")

    try:
        pred = model.predict(X)
        st.success(f"Prediction: {pred[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Show data preview
if df is not None:
    with st.expander("Show dataset (employee_clean_data.csv)"):
        st.dataframe(df.head(200))
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV preview",
            data=csv,
            file_name="employee_clean_data_preview.csv",
            mime="text/csv",
        )
