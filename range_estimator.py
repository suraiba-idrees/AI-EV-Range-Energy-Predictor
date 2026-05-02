import os
import pandas as pd
import numpy as np
import joblib

# 1. Load trained artifacts (Updated for Ensemble)
script_dir = os.path.dirname(os.path.abspath(__file__))
ensemble_model = joblib.load(os.path.join(script_dir, "ev_model.pkl"))
encoder = joblib.load(os.path.join(script_dir, "encoder.pkl"))
scaler = joblib.load(os.path.join(script_dir, "scaler.pkl"))
model_columns = joblib.load(os.path.join(script_dir, "model_columns.pkl"))

# Input helper functions
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
        print(f"Invalid input! Choose from: {options}")

print("--- Professional EV Range Estimator ---")

# 2. Collect user input
new_trip = {
    'Speed_kmh': get_float_input("Speed (km/h)", 5),
    'Battery_State_%': get_float_input("Current Battery (%)", 0, 100),
    'Battery_Voltage_V': get_float_input("Battery Voltage (V)", 1), # Added to match training
    'Battery_Temperature_C': get_float_input("Battery Temp (°C)"),
    'Slope_%': get_float_input("Avg Road Slope (%)"),
    'Temperature_C': get_float_input("Ambient Temp (°C)"),
    'Humidity_%': get_float_input("Humidity (%)", 0, 100), # Added to match training
    'Wind_Speed_ms': get_float_input("Wind Speed (m/s)", 0), # Added to match training
    'Tire_Pressure_psi': get_float_input("Tire Pressure (psi)", 10), # Added to match training
    'Vehicle_Weight_kg': get_float_input("Total Weight (kg)", 500),
    'Distance_Travelled_km': get_float_input("Reference Distance (km)", 0.1),
    'Driving_Mode': get_str_input("Driving Mode", ['Eco', 'Normal', 'Sport']),
    'Road_Type': get_str_input("Road Type", ['Highway', 'City', 'Rural']),
    'Traffic_Condition': get_str_input("Traffic", ['Low', 'Medium', 'High']),
    'Weather_Condition': get_str_input("Weather", ['Clear', 'Rain', 'Snow', 'Fog'])
}

# Extra Parameters
battery_capacity_kwh = get_float_input("Battery Total Capacity (kWh)", 10)
driving_style = get_str_input("Driving Style (Conservative/Aggressive)", ['Conservative', 'Aggressive'])
hvac_on = get_str_input("Is Climate Control ON? (y/n)", ['y', 'n'])

# 3. Data Processing (Scaling & Encoding)
def process_for_model(data_dict):
    df = pd.DataFrame([data_dict])
    cat_cols = ['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']
    num_cols = [c for c in df.columns if c not in cat_cols]
    
    # Critical Scaling
    df[num_cols] = scaler.transform(df[num_cols])
    # Encoding
    enc = encoder.transform(df[cat_cols])
    enc_df = pd.DataFrame(enc, columns=encoder.get_feature_names_out(cat_cols))
    
    final = pd.concat([df[num_cols].reset_index(drop=True), enc_df], axis=1)
    return final.reindex(columns=model_columns, fill_value=0)

model_input = process_for_model(new_trip)

# 4. Predict Base Consumption (with expm1 for real kWh)
log_pred = ensemble_model.predict(model_input)[0]
base_energy_mean = np.expm1(log_pred)

# 5. Domain Logic Adjustments
# Cold-start
if new_trip['Battery_Temperature_C'] < 10:
    base_energy_mean *= 1.15

# Driving style
style_mult = 1.15 if driving_style == 'Aggressive' else 1.0
base_energy_mean *= style_mult

# HVAC Load
hvac_power_kw = 0.0
if hvac_on == 'y':
    temp = new_trip['Temperature_C']
    if temp < 5: hvac_power_kw = 5.0
    elif temp < 15: hvac_power_kw = 2.0
    elif temp > 30: hvac_power_kw = 3.5
    else: hvac_power_kw = 1.0

trip_time_hours = new_trip['Distance_Travelled_km'] / new_trip['Speed_kmh']
hvac_energy_total = hvac_power_kw * trip_time_hours
total_energy_needed = base_energy_mean + hvac_energy_total

# 6. Battery & Range Calculation
# 5% Safety Reserve
usable_kwh = (battery_capacity_kwh * new_trip['Battery_State_%'] / 100) * 0.95

# Efficiency = Total Energy used / Distance
efficiency = total_energy_needed / new_trip['Distance_Travelled_km']
estimated_range = usable_kwh / efficiency

# 7. Professional Output
print("\n" + "="*45)
print(" 🔋 EV RANGE ESTIMATE (Production Grade)")
print("="*45)
print(f"ESTIMATED RANGE:    {estimated_range:.1f} km")
print(f"AVG EFFICIENCY:     {efficiency:.3f} kWh/km")
print(f"Climate Impact:     {hvac_energy_total:.2f} kWh added")
print(f"Battery Reserved:   5% Safety Buffer active")
print("="*45)

if estimated_range < 20:
    print("🚨 WARNING: Find a charger immediately!")