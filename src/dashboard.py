"""Simple Streamlit dashboard to explore dataset and run single predictions."""
import streamlit as st
import pandas as pd
from pathlib import Path
import joblib


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


DATA_PATH = Path("data/synthetic_heart_disease_dataset.csv")
MODEL_PATH = Path("models/best_model.joblib")


def main():
    st.title("Heart Disease Prototype Dashboard")
    df = load_data(DATA_PATH)

    st.sidebar.header("Quick actions")
    if st.sidebar.button("Show raw data sample"):
        st.dataframe(df.sample(10))

    st.header("Target distribution")
    if "Heart_Disease" in df.columns:
        st.bar_chart(df["Heart_Disease"].value_counts())

    st.header("Single-patient prediction")
    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH)
        st.write("Enter patient features (minimal):")
        with st.form("predict_form"):
            Age = st.number_input("Age", min_value=18, max_value=120, value=60)
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Weight = st.number_input("Weight (kg)", value=80)
            Height = st.number_input("Height (cm)", value=170)
            BMI = st.number_input("BMI", value=26.0)
            Systolic_BP = st.number_input("Systolic BP", value=140)
            Diastolic_BP = st.number_input("Diastolic BP", value=90)
            Cholesterol_Total = st.number_input("Cholesterol Total", value=220)
            submitted = st.form_submit_button("Predict")

        if submitted:
            row = pd.DataFrame([{"Age": Age, "Gender": Gender, "Weight": Weight, "Height": Height, "BMI": BMI, "Systolic_BP": Systolic_BP, "Diastolic_BP": Diastolic_BP, "Cholesterol_Total": Cholesterol_Total}])
            try:
                score = float(model.predict_proba(row)[:, 1][0])
                st.metric("Risk score", f"{score:.3f}")
                st.write("Label:", "High" if score >= 0.7 else ("Medium" if score >= 0.4 else "Low"))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.warning("Model not found. Please run training first.")


if __name__ == "__main__":
    main()
