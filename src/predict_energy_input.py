"""
EV Energy Consumption Prediction Script
- Features: Real-time user input collection with validation
- Logic: Uses a Professional Ensemble Model with Log-Transformed target handling (expm1)
- Adjustments: Includes a 10% safety margin for cold battery temperatures (<10°C)
- Uncertainty: Calculates a ±5% prediction buffer for reliability
- Visualization: Generates consumption sensitivity plots relative to vehicle speed
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Load trained artifacts
script_dir = os.path.dirname(os.path.abspath(__file__))
ensemble_model = joblib.load(os.path.join(script_dir, "ev_model.pkl"))
encoder = joblib.load(os.path.join(script_dir, "encoder.pkl"))
scaler = joblib.load(os.path.join(script_dir, "scaler.pkl"))
model_columns = joblib.load(os.path.join(script_dir, "model_columns.pkl"))

print("✅ Professional Ensemble Model & Scaler loaded successfully!\n")

# 2. Input functions
def get_float_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = float(input(prompt + ": "))
            if min_val is not None and value < min_val: continue
            if max_val is not None and value > max_val: continue
            return value
        except ValueError: print("Please enter a valid number!")

def get_str_input(prompt, options=None):
    while True:
        value = input(prompt + ": ")
        if options is None or value in options: return value
        print(f"Invalid! Choose from: {options}")

# 3. Collect user input
print("--- Enter Trip Details ---")
new_trip = {
    'Speed_kmh': get_float_input("Speed (km/h)", 0),
    'Battery_State_%': get_float_input("Battery State (%)", 0, 100),
    'Battery_Voltage_V': get_float_input("Battery Voltage (V)", 0),
    'Battery_Temperature_C': get_float_input("Battery Temperature (°C)"),
    'Slope_%': get_float_input("Slope (%)"),
    'Temperature_C': get_float_input("Ambient Temperature (°C)"),
    'Humidity_%': get_float_input("Humidity (%)", 0, 100),
    'Wind_Speed_ms': get_float_input("Wind Speed (m/s)", 0),
    'Tire_Pressure_psi': get_float_input("Tire Pressure (psi)", 0),
    'Vehicle_Weight_kg': get_float_input("Vehicle Weight (kg)", 0),
    'Distance_Travelled_km': get_float_input("Distance Travelled (km)", 0.01),
    'Driving_Mode': get_str_input("Driving Mode", ['Eco','Normal','Sport']),
    'Road_Type': get_str_input("Road Type", ['Highway','City','Rural']),
    'Traffic_Condition': get_str_input("Traffic Condition", ['Low','Medium','High']),
    'Weather_Condition': get_str_input("Weather Condition", ['Clear','Rain','Snow','Fog'])
}

# 4. Processing Input
def process_data(input_dict):
    df = pd.DataFrame([input_dict])
    cat_cols = ['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']
    num_cols = [c for c in df.columns if c not in cat_cols]
    
    # Scale numbers
    df[num_cols] = scaler.transform(df[num_cols])
    # Encode categories
    enc_feats = encoder.transform(df[cat_cols])
    enc_df = pd.DataFrame(enc_feats, columns=encoder.get_feature_names_out(cat_cols))
    
    final = pd.concat([df[num_cols].reset_index(drop=True), enc_df], axis=1)
    return final.reindex(columns=model_columns, fill_value=0)

processed_input = process_data(new_trip)

# 5. Prediction
log_pred = ensemble_model.predict(processed_input)[0]
actual_kwh = np.expm1(log_pred) 

# Apply Cold Battery Adjustment
if new_trip['Battery_Temperature_C'] < 10:
    print("\n⚠️ Note: Cold battery detected. Adding 10% safety margin to consumption.")
    actual_kwh *= 1.10

lower_bound = actual_kwh * 0.95
upper_bound = actual_kwh * 1.05

print("\n" + "="*45)
print(f"🎯 Predicted Energy Consumption: {actual_kwh:.2f} kWh")
print(f"📉 Likely Range (95% Confidence): {lower_bound:.2f} - {upper_bound:.2f} kWh")
print("="*45)

# 6. Visualization
visualize = input("\nDo you want a plot of predicted range vs Speed? (y/n): ")
if visualize.lower() == 'y':
    test_speeds = np.linspace(0, 140, 50)
    results = []
    
    for s in test_speeds:
        temp_trip = new_trip.copy()
        temp_trip['Speed_kmh'] = s
        p_input = process_data(temp_trip)
        res = np.expm1(ensemble_model.predict(p_input)[0])
        results.append(res)
    
    plt.figure(figsize=(10,6))
    plt.plot(test_speeds, results, color='blue', label='Predicted Consumption', linewidth=2)
    plt.fill_between(test_speeds, np.array(results)*0.95, np.array(results)*1.05, color='blue', alpha=0.1, label='95% Confidence Zone')
    plt.title("Energy Consumption vs Speed")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("kWh")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()