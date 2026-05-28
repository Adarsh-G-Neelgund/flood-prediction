import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Flood Predictor", page_icon="🌊", layout="centered")
st.title("🌊 Smart Flood Risk Prediction System")
st.write("Enter localized data points to calculate the mathematical probability of a flood.")


@st.cache_resource
def load_pipeline():
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('selector.pkl')
    pca = joblib.load('pca.pkl')
    dummy_columns = joblib.load('dummy_columns.pkl')
    return model, scaler, selector, pca, dummy_columns

model, scaler, selector, pca, dummy_columns = load_pipeline()


st.header("📊 Input Parameters")
col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude", value=18.8)
    longitude = st.number_input("Longitude", value=78.8)
    rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 120.0)
    temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    river_discharge = st.number_input("River Discharge (m³/s)", value=2000.0)
    water_level = st.slider("Water Level (m)", 0.0, 20.0, 4.5)

with col2:
    elevation = st.number_input("Elevation (m)", value=300.0)
    land_cover = st.selectbox("Land Cover Type", ['Water Body', 'Forest', 'Agricultural', 'Desert', 'Urban'])
    soil_type = st.selectbox("Soil Type", ['Clay', 'Peat', 'Loam', 'Sandy', 'Silt'])
    population_density = st.number_input("Population Density", value=5000.0)
    infrastructure = st.selectbox("Infrastructure Quality", [0, 1], format_func=lambda x: "Good/Maintained (1)" if x==1 else "Poor/Substandard (0)")
    historical_floods = st.selectbox("Historically Flooded?", [0, 1], format_func=lambda x: "Yes (1)" if x==1 else "No (0)")

if st.button(" Predict ", type="primary", use_container_width=True):
    
    input_data = {
        'Latitude': latitude, 'Longitude': longitude, 'Rainfall_mm': rainfall,
        'Temperature_C': temperature, 'Humidity': humidity, 'River_Discharge_m3_s': river_discharge,
        'Water_Level_m': water_level, 'Elevation_m': elevation, 'Land_Cover': land_cover,
        'Soil_Type': soil_type, 'Population_Density': population_density,
        'Infrastructure': infrastructure, 'Historical_Floods': historical_floods
    }
    
    
    input_df = pd.DataFrame([input_data])
    
    
    input_encoded = pd.get_dummies(input_df)
    
    
    input_final = input_encoded.reindex(columns=dummy_columns, fill_value=0)
    
    
    scaled_data = scaler.transform(input_final)
    selected_data = selector.transform(scaled_data)
    extracted_data = pca.transform(selected_data)
    
    
    probability = model.predict_proba(extracted_data)[0][1]
    prediction = model.predict(extracted_data)[0]
    
    
    st.subheader("🔮 Pipeline Results")
    if prediction == 1:
        st.error(f"🔴 **CRITICAL RISK:** Flood predicted! Probability: **{probability * 100:.2f}%**")
    else:
        st.success(f"🟢 **LOW RISK:** Area clear. Flood probability is only **{probability * 100:.2f}%**")