import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from pathlib import Path

st.set_page_config(page_title="Model Predictor", layout="wide")

st.title("📊 Streamlit prediction app — Student_model.pkl")

# --- File paths (change if your files are in a different location) ---
DEFAULT_MODEL_PATH = Path("Student_model.pkl")
DEFAULT_DATA_PATH = Path("employee_clean_data.csv")

# --- Helpers ---
@st.cache_data
def load_model(path: Path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data(path: Path):
    return pd.read_csv(path)

def infer_feature_names(model, df):
    """
    Try several strategies to find expected feature names:
      1. model.feature_names_in_ (sklearn)
      2. If model is a pipeline, try final estimator attribute
      3. Look for saved 'feature_names' attribute
      4. Fallback to df.columns (drop common target column names)
    """
    # 1. sklearn feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # 2. pipeline case: try final estimator
    try:
        # many pipelines have named_steps or steps
        if hasattr(model, "named_steps"):
            last = list(model.named_steps.items())[-1][1]
            if hasattr(last, "feature_names_in_"):
                return list(last.feature_names_in_)
        if hasattr(model, "steps"):
            last = model.steps[-1][1]
            if hasattr(last, "feature_names_in_"):
                return list(last.feature_names_in_)
    except Exception:
        pass

    # 3. custom attribute
    if hasattr(model, "feature_names"):
        return list(getattr(model, "feature_names"))

    # 4. fallback: use df columns minus likely label columns
    if df is not None:
        cols = list(df.columns)
        # drop common label names if present
        for t in ["target", "label", "y", "Outcome", "result", "Result"]:
            if t in cols:
                cols.remove(t)
        return cols

    return []

def make_input_widget(col, sample_series):
    """Create appropriate input widget depending on dtype / unique values"""
    if pd.api.types.is_numeric_dtype(sample_series):
        min_v = float(sample_series.min()) if not np.isnan(sample_series.min()) else 0.0
        max_v = float(sample_series.max()) if not np.isnan(sample_series.max()) else min_v + 100.0
        mean_v = float(sample_series.median() if not np.isnan(sample_series.median()) else (min_v + max_v) / 2)
        return st.number_input(col, value=mean_v, format="%.6f")
    else:
        uniques = sample_series.dropna().unique().tolist()
        if 1 <= len(uniques) <= 20:
            # provide selectbox for small cardinalities
            return st.selectbox(col, options=[""] + uniques, index=0)
        else:
            return st.text_input(col, value="")

# --- Sidebar: allow user to pick alternate files if needed ---
st.sidebar.header("Files & options")
model_file = st.sidebar.file_uploader("Upload a .pkl model (optional)", type=["pkl"])
data_file = st.sidebar.file_uploader("Upload a CSV data file (optional)", type=["csv"])

use_defaults = False
if model_file is None and data_file is None:
    use_defaults = True
    st.sidebar.markdown("Using default files in app directory:")
    st.sidebar.text(f"Model: {DEFAULT_MODEL_PATH}")
    st.sidebar.text(f"Data: {DEFAULT_DATA_PATH}")

# Load model
try:
    if model_file is not None:
        model = pickle.load(model_file)
    else:
        if not DEFAULT_MODEL_PATH.exists():
            st.error(f"Default model not found at `{DEFAULT_MODEL_PATH}`. Please upload a model file or place it in the app folder.")
            st.stop()
        model = load_model(DEFAULT_MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load data (for sample values / dynamic inputs)
df = None
try:
    if data_file is not None:
        data_bytes = data_file.read()
        data_stream = io.BytesIO(data_bytes)
        df = pd.read_csv(data_stream)
    else:
        if DEFAULT_DATA_PATH.exists():
            df = load_data(DEFAULT_DATA_PATH)
        else:
            st.warning("No data CSV found. The app will still try to infer feature names from the model.")
            df = None
except Exception as e:
    st.warning(f"Failed to load CSV (used for input defaults): {e}")
    df = None

# Infer feature names
feature_names = infer_feature_names(model, df)
if not feature_names:
    st.error("Couldn't infer feature names from the model or CSV. Please upload a CSV containing example rows or a model with feature metadata.")
    st.stop()

st.sidebar.markdown(f"Detected **{len(feature_names)}** features")

# Main: build input form dynamically
st.subheader("Input features")
with st.form("predict_form"):
    input_values = {}
    # If df available, use its column data to derive widget types and defaults
    for col in feature_names:
        sample_series = df[col] if (df is not None and col in df.columns) else pd.Series(dtype="float")
        input_values[col] = make_input_widget(col, sample_series)
    submit = st.form_submit_button("Predict")

# Perform prediction
if submit:
    # Build single-row DataFrame in the same column order
    X = pd.DataFrame([input_values], columns=feature_names)
    st.write("### Input preview")
    st.dataframe(X)

    # Try to predict
    try:
        pred = model.predict(X)
        st.success(f"Prediction: {pred[0]}")
        # If probability available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # Show class probabilities
            if probs.shape[1] > 1:
                proba_df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in model.classes_])
                st.write("### Predicted probabilities")
                st.dataframe(proba_df.T)
            else:
                st.write(f"Probability (single-output): {probs[0,0]:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Show dataset on demand
st.sidebar.markdown("---")
if df is not None:
    show_data = st.sidebar.checkbox("Show loaded CSV data", value=False)
    if show_data:
        st.subheader("Dataset preview")
        st.dataframe(df.head(200))

        # allow download of displayed dataframe (csv)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="employee_clean_data_preview.csv", mime="text/csv")
else:
    st.info("No CSV loaded to preview. Upload one via the sidebar to preview sample rows.")

st.sidebar.markdown("---")
st.sidebar.markdown("Notes:")
st.sidebar.markdown(
    "- If categorical variables were label-encoded during model training, supply the encoded numeric values in inputs OR upload a CSV with the same preprocessing.\n"
    "- If your model expects features in a special transformed format (e.g., one-hot columns), ensure you provide the same column names in the CSV used here."
)

st.write("---")
st.write("If you want, upload a different model or CSV from the sidebar to test other models/datasets.")
