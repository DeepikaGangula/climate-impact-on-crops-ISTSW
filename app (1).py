import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("ml_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌾 Climate Impact Analysis on Crop Yield")

st.write("Predict crop yield based on climate conditions")

# User input
temperature = st.number_input("Enter Temperature")
rainfall = st.number_input("Enter Rainfall")

if st.button("Predict Yield"):
    input_data = np.array([[temperature, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Yield: {prediction[0]:.2f}")
