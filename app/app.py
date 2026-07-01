# ============================================
# STEP 10: DEPLOYMENT - STREAMLIT APP
# Road Accident Severity Predictor
# Model: Random Forest (Best Model)
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load(r"C:\Users\vaibh\OneDrive\Desktop\road_accident_project\models\best_model.pkl")
scaler = joblib.load(r"C:\Users\vaibh\OneDrive\Desktop\road_accident_project\models\scaler.pkl")

st.title("Road Accident Severity Predictor")
st.write("Fill in the details below to predict accident severity.")

day_of_week = st.selectbox("Day of Week", [1,2,3,4,5,6,7])
speed_limit = st.selectbox("Speed Limit", [20,30,40,50,60,70])
hour = st.slider("Hour of Day", 0, 23, 12)
number_of_vehicles = st.slider("Number of Vehicles", 1, 10, 1)
number_of_casualties = st.slider("Number of Casualties", 1, 10, 1)
urban_or_rural = st.selectbox("Area Type", [1,2], format_func=lambda x: "Urban" if x==1 else "Rural")

if st.button("Predict"):
    columns = scaler.feature_names_in_
    input_dict = {col: 0 for col in columns}
    
    input_dict['Day_of_Week'] = day_of_week
    input_dict['Speed_limit'] = speed_limit
    input_dict['Hour'] = hour
    input_dict['Number_of_Vehicles'] = number_of_vehicles
    input_dict['Number_of_Casualties'] = number_of_casualties
    input_dict['Urban_or_Rural_Area'] = urban_or_rural

    input_data = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    severity_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
    result = severity_map[int(prediction)]

    if result == "Fatal":
        st.error("Fatal Accident")
    elif result == "Serious":
        st.warning("Serious Accident")
    else:
        st.success("Slight Accident")