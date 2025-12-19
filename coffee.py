import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.write(model.feature_names_in_)


st.title("☕ Coffee Shop Daily Revenue Prediction")
st.write("Enter the details below to predict Daily Revenue")

# ===== USER INPUTS =====
location_foot_traffic = st.number_input(
    "Location Foot Traffic (people per day)", min_value=0
)

marketing_spend = st.number_input(
    "Marketing Spend Per Day (₹)", min_value=0.0
)

num_customers = st.number_input(
    "Number of Customers Per Day", min_value=0
)

avg_order_value = st.number_input(
    "Average Order Value (₹)", min_value=0.0
)

num_employees = st.number_input(
    "Number of Employees", min_value=0
)

operating_hours = st.number_input(
    "Operating Hours Per Day", min_value=0.0
)

# ===== PREDICTION =====
if st.button("Predict Revenue"):
    # Step 1: create dataframe (order doesn't matter here)
    input_data = pd.DataFrame({
        "Location_Foot_Traffic": [location_foot_traffic],
        "Marketing_Spend_Per_Day": [marketing_spend],
        "Number_of_Customers_Per_Day": [num_customers],
        "Average_Order_Value": [avg_order_value],
        "Number_of_Employees": [num_employees],
        "Operating_Hours_Per_Day": [operating_hours]
    })

    # Step 2: FORCE SAME ORDER AS TRAINING (🔥 FIX)
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Daily Revenue: ₹ {prediction[0]:.2f}")
