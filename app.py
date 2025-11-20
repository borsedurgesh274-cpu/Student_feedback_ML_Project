import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- Configuration ---
MODEL_FILE = 'Student_model.pkl'

# --- Custom Styling for better UI ---
st.set_page_config(
    page_title="Employee Feedback Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main-header {
        color: #1E90FF; /* Dodger Blue */
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stSidebar .stSelectbox {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)


# --- Functions to Load Resources ---

@st.cache_resource
def load_model():
    """Loads the pickled model file."""
    if not os.path.exists(MODEL_FILE):
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure it is uploaded.")
        return None
    try:
        with open(MODEL_FILE, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# --- Main App Logic ---

st.markdown('<div class="main-header">Employee Feedback Predictor</div>', unsafe_allow_html=True)
st.write("This application predicts employee feedback (Average, Good, or Poor) based on input features using a pre-trained machine learning model.")

if model is not None:
    # --- Sidebar for Input Features ---
    st.sidebar.header("Input Employee Data")

    # 1. Age (Assuming a reasonable range)
    age = st.sidebar.slider("Age (Years)", min_value=18, max_value=65, value=30, step=1)

    # 2. Experience (Assuming a reasonable range)
    experience = st.sidebar.slider("Experience (Years)", min_value=0, max_value=40, value=5, step=1)

    # 3. Salary (Assuming a general range, using number input for flexibility)
    salary = st.sidebar.number_input("Annual Salary ($)", min_value=20000, max_value=200000, value=75000, step=1000)

    # 4. Department (Mapping the likely encoded integers to user-friendly labels)
    department_map = {
        'Engineering (0)': 0,
        'Sales (1)': 1,
        'Marketing (2)': 2,
        'HR (3)': 3,
        'Finance (4)': 4
    }
    department_label = st.sidebar.selectbox(
        "Department Code",
        options=list(department_map.keys()),
        index=0
    )
    department = department_map[department_label]

    # Combine inputs into a DataFrame for the model
    input_data = pd.DataFrame({
        'Age': [age],
        'Experience': [experience],
        'Salary': [salary],
        'Department': [department]
    })

    st.subheader("Input Summary")
    st.dataframe(input_data, use_container_width=True)


    # --- Prediction ---
    if st.button("Predict Employee Feedback"):
        try:
            # The model predicts the class
            prediction_result = model.predict(input_data)[0]
            
            # The model's classes (from inspection) are typically ['Average', 'Good', 'Poor']
            # We'll use a dictionary to color-code the output
            color_map = {
                'Good': 'green',
                'Average': 'orange',
                'Poor': 'red'
            }
            
            # Get the appropriate color
            result_color = color_map.get(prediction_result, 'blue')

            st.markdown("---")
            st.subheader("Prediction Result")

            st.markdown(
                f"""
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border-left: 5px solid {result_color};">
                    The predicted Employee Feedback is: <span style="color: {result_color}; font-size: 1.5em; font-weight: bold;">{prediction_result}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")


        except Exception as e:
            st.error(f"An error occurred during prediction. Please check model compatibility: {e}")

else:
    st.warning("Prediction cannot be performed because the model failed to load.")

# --- Data File Display (Optional but helpful for context) ---

st.sidebar.markdown("---")
st.sidebar.subheader("Data Reference")

# Check if the CSV file exists (the user uploaded it, so it should be in the same directory)
DATA_FILE = 'employee_clean_data.csv'
if os.path.exists(DATA_FILE):
    st.sidebar.info(f"The original data file (`{DATA_FILE}`) is available for context.")
    if st.sidebar.checkbox('Show first 5 rows of data'):
        try:
            df = pd.read_csv(DATA_FILE)
            st.subheader("First 5 Rows of Training Data")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load data file: {e}")
else:
    st.sidebar.warning(f"Data file '{DATA_FILE}' not found.")
