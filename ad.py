import streamlit as st
import numpy as np
import joblib

# Load the joblib logistic model
model = joblib.load("log_reg_model")

st.title("Ad Click Prediction App (Logistic Regression)")
st.write("Enter the details below to predict whether the user will click on the ad.")

# Inputs
daily_time = st.number_input("Daily Time Spent on Site", min_value=0.0, step=0.1)
age = st.number_input("Age", min_value=0, step=1)
income = st.number_input("Area Income", min_value=0.0, step=100.0)
internet_usage = st.number_input("Daily Internet Usage", min_value=0.0, step=0.1)
gender = st.selectbox("Gender", ["Female", "Male"])

male = 1 if gender == "Male" else 0

# Prepare input array (same order as training)
input_data = np.array([[daily_time, age, income, internet_usage, male]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write("----")
    st.write(f"**Probability of Click (1): {prob:.2f}**")

    if prediction == 1:
        st.success("User is **likely to click** on the ad.")
    else:
        st.error("User is **unlikely to click** on the ad.")
