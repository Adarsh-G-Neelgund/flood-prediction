import streamlit as st
import pickle  
import pandas as pd
import numpy as np

st.set_page_config(page_title="Flood Predictor", layout="centered") 
st.title(" 🌊Flood Prediction System ")
st.markdown("""
This interactive web application uses **Machine Learning** to calculate the statistical probability of a flood 
based on live weather and environmental conditions. 

**How to use:** Move the sliders below to simulate different weather scenarios (like heavy rainfall or changing river levels), 
and the trained AI model will instantly analyze the parameters to predict the danger level in real-time.
""")

@st.cache_resource
def load_pipeline():
    with open('logistic_regression_model.pkl', 'rb') as f: model = pickle.load(f)
    with open('baseline_model.pkl', 'rb') as f: model_base = pickle.load(f) 
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    with open('selector.pkl', 'rb') as f: selector = pickle.load(f)
    with open('pca.pkl', 'rb') as f: pca = pickle.load(f)
    with open('dummy_columns.pkl', 'rb') as f: dummy_columns = pickle.load(f)
    return model, model_base, scaler, selector, pca, dummy_columns

model, model_base, scaler, selector, pca, dummy_columns = load_pipeline()

st.header("Input Parameters")
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

if st.button("Predict", type="primary", use_container_width=True):
    
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
    
    # ---- 🏆 Pipeline A: Optimized Model (Logistic Regression) ----
    scaled_data = scaler.transform(input_final)
    selected_data = selector.transform(scaled_data)
    extracted_data = pca.transform(selected_data)
    probability_lr = model.predict_proba(extracted_data)[0][1]
    prediction_lr = model.predict(extracted_data)[0]
    
    # ---- 📉 Pipeline B: Baseline Model (Naïve Bayes) ----
    probability_nb = model_base.predict_proba(input_final)[0][1]
    prediction_nb = model_base.predict(input_final)[0]
    
    # =========================================================
    # 📈 DASHBOARD RENDERING: MULTI-ALGORITHM SIDE-BY-SIDE
    # =========================================================
    st.markdown("---")
    st.header("📊 Multi-Algorithm Risk Assessment Dashboard")
    
    col_ui_lr, col_ui_base = st.columns(2)
    
    with col_ui_lr:
        st.subheader("🏆 Model A: Logistic Regression Pipeline")
        st.caption("Our Optimized Production Model (With Scaling & PCA Component Reduction)")
        
        risk_lr = probability_lr * 100
        if prediction_lr == 1:
            st.error(f"🚨 **CRITICAL RISK:** Flood predicted! Probability: **{risk_lr:.2f}%**")
        else:
            st.success(f"🟢 **LOW RISK:** Area clear. Flood probability: **{risk_lr:.2f}%**")
            
        st.markdown("""
        | Validation Performance Metrics | Score Summary |
        | :--- | :--- |
        | **Overall Model Accuracy** | **85.45%** |
        | **Precision Score** | **85.00%** |
        | **Recall (Sensitivity)** | **80.00%** |
        """)

    with col_ui_base:
        st.subheader("📉 Model B: Naïve Bayes Baseline")
        st.caption("Standard Classifier Blueprint (Raw Unscaled Fields, No Feature Compressions)")
        
        risk_nb = probability_nb * 100
        if prediction_nb == 1:
            st.error(f"⚠️ **HAZARD WARNING:** Threat detected! Expected risk score: **{risk_nb:.2f}%**")
        else:
            st.success(f"✅ **STABLE AREA:** Low baseline movement. Score: **{risk_nb:.2f}%**")
            
        st.markdown("""
        | Baseline Performance Metrics | Score Summary |
        | :--- | :--- |
        | *Overall Model Accuracy* | *72.35%* |
        | *Precision Score* | *68.10%* |
        | *Recall (Sensitivity)* | *64.40%* |
        """)
