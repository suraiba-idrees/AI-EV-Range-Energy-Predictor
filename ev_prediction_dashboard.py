import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go

# --- 1. CONFIG & REFINED UI ---
st.set_page_config(page_title="NEURA-DRIVE | AI EV Engine", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Inter:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1a2a3a 0%, #0a0e14 80%);
        background-attachment: fixed;
        color: #e2e8f0;
    }

    /* REAL FROSTED GLASS SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    .main-title {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(to bottom, #fff 30%, #00f2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 8px;
    }

    .telemetry-tag {
        font-family: 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        letter-spacing: 4px;
        color: #64748b;
        text-align: center;
        display: block;
        margin-top: -10px;
        font-size: 0.9rem;
    }

    /* Professional Metric Headings */
    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        text-align: center;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1.5px;
        font-size: 0.85rem;
    }

    /* SINGLE BLOCK AI INSIGHT CONTAINER */
    .ai-unified-box {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 35px;
        border: 1px solid rgba(0, 242, 255, 0.15);
        margin: 20px 0;
        display: flex;
        justify-content: space-between;
        gap: 40px;
    }

    .insight-section {
        flex: 1;
    }

    .ai-header {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 1.2px;
        margin-bottom: 12px;
        text-transform: uppercase;
    }

    .divider {
        width: 1px;
        background: rgba(255, 255, 255, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(base_path, "ev_model.pkl"))
        scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
        encoder = joblib.load(os.path.join(base_path, "encoder.pkl"))
        model_columns = joblib.load(os.path.join(base_path, "model_columns.pkl"))
        csv_path = os.path.join(base_path, "EV_dataset.csv")
        if not os.path.exists(csv_path): csv_path = os.path.join(base_path, "Data", "EV_dataset.csv")
        return model, scaler, encoder, model_columns, pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Missing Files: {e}"); st.stop()

model, scaler, encoder, model_columns, ev_data = load_assets()

# --- 3. SIDEBAR (Original Icons & Titles) ---
st.sidebar.markdown("<h2 style='color:#00f2ff; font-family:Orbitron; font-size:1.2rem;'>🏎️ COCKPIT CONTROLS</h2>", unsafe_allow_html=True)
with st.sidebar.expander("🔋 Battery System", expanded=True):
    capacity = st.number_input("Total Capacity (kWh)", 10, 150, 75)
    soc = st.slider("State of Charge (%)", 5, 100, 85)
    weight = st.number_input("Vehicle Weight (kg)", 1000, 3000, 1800)
with st.sidebar.expander("🛣️ Mission Parameters", expanded=True):
    speed = st.slider("Target Speed (km/h)", 20, 160, 90)
    dist_planned = st.number_input("Trip Distance (km)", 1, 500, 50)
    mode = st.selectbox("Drive Mode", ["Normal", "Eco", "Sport"])
    style = st.radio("Drive Profile", ["Conservative", "Aggressive"], horizontal=True)
with st.sidebar.expander("🌍 Environment", expanded=False):
    road = st.selectbox("Road", ["Highway", "City", "Rural"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Fog"])
    temp = st.slider("Ambient Temp (°C)", -15, 45, 25)
    hvac = st.toggle("HVAC Optimization", value=True)

# --- 4. ENGINE LOGIC ---
def run_prediction():
    input_row = ev_data.mean(numeric_only=True).to_dict()
    input_row.update({'Speed_kmh': speed, 'Battery_State_%': soc, 'Temperature_C': temp,
        'Vehicle_Weight_kg': weight, 'Distance_Travelled_km': dist_planned,
        'Road_Type': road, 'Traffic_Condition': traffic, 'Weather_Condition': weather, 'Driving_Mode': mode})
    
    df = pd.DataFrame([input_row])
    num_cols = list(scaler.feature_names_in_)
    df[num_cols] = scaler.transform(df[num_cols])
    encoded = encoder.transform(df[['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']))
    final_x = pd.concat([df[num_cols].reset_index(drop=True), encoded_df], axis=1).reindex(columns=model_columns, fill_value=0)
    
    log_pred = model.predict(final_x)
    base_energy = np.expm1(log_pred)
    
    if style == "Aggressive": base_energy *= 1.15
    if temp < 10: base_energy *= 1.10
    hvac_total = (4.0 if temp < 5 else 3.0 if temp > 30 else 1.0) * (dist_planned / speed) if hvac else 0
    total_energy = base_energy + hvac_total
    estimated_range = ((capacity * soc / 100) * 0.95) / (total_energy / dist_planned)
    
    try:
        rf_submodel = model.estimators_[0] 
        explainer = shap.TreeExplainer(rf_submodel)
        shap_vals = explainer.shap_values(final_x)
    except:
        explainer = shap.Explainer(model.predict, final_x)
        shap_vals = explainer(final_x).values
    
    return estimated_range, total_energy, shap_vals, final_x

def create_gauge(value, unit, max_val, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        number = {'suffix': unit, 'font': {'size': 55, 'family': 'Space Grotesk', 'color': 'white'}},
        gauge = {
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "rgba(0,0,0,0)"},
            'bar': {'color': color, 'thickness': 0.2},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
        }
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# --- 5. MAIN DASHBOARD ---
st.markdown("<div class='main-title'>NEURA DRIVE</div>", unsafe_allow_html=True)
st.markdown("<span class='telemetry-tag'>◌ Advanced Propulsion Analytics ◌</span>", unsafe_allow_html=True)

if st.sidebar.button("🚀 INITIATE PREDICTION", use_container_width=True):
    with st.status("🔗 Syncing with Vehicle Neural Core...", expanded=False) as status:
        time.sleep(0.5)
        status.update(label="✅ Computation Complete", state="complete")

    range_km, energy, shap_v, processed_df = run_prediction()
    r_val = float(range_km[0]) if isinstance(range_km, (np.ndarray, list)) else float(range_km)
    e_val = float(energy[0]) if isinstance(energy, (np.ndarray, list)) else float(energy)

    # Gauges with Professional Labels
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<p class='metric-label'>Estimated Range</p>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(r_val, " KM", 600, "#00f2ff"), use_container_width=True)
    with c2:
        st.markdown("<p class='metric-label'>Energy Consumption</p>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge(e_val, " kWh", 100, "#ff4b4b"), use_container_width=True)

    st.markdown("<h3 style='font-family:Space Grotesk; font-size:1.3rem; margin-top:20px; font-weight:600;'>🧠 Neural Core Insights</h3>", unsafe_allow_html=True)
    
    shap_abs = np.abs(shap_v[0]) if len(shap_v.shape) > 1 else np.abs(shap_v)
    top_feature = model_columns[np.argmax(shap_abs)].replace('_', ' ')
    
    # UNIFIED AI INSIGHT BLOCK (Like Reference)
    st.markdown(f"""
    <div class="ai-unified-box">
        <div class="insight-section">
            <p class="ai-header" style="color:#00f2ff;">Core Rationale</p>
            <p style="color:#cbd5e1; font-size:1.05rem; line-height:1.6;">
                AI processing identified <b style="color:white;">{top_feature.upper()}</b> as the primary energy consumption catalyst for this session.
            </p>
        </div>
        <div class="divider"></div>
        <div class="insight-section">
            <p class="ai-header" style="color:#10b981;">Strategic Optimization</p>
            <p style="color:#cbd5e1; font-size:1.05rem; line-height:1.6;">
                Reducing current velocity by 15% would extend mission endurance to <b style="color:#10b981;">{r_val*1.15:.1f} KM</b>.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 XAI Feature Interpretability Matrix", expanded=False):
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_alpha(0); ax.set_facecolor('none')
        s_plot_val = shap_v[0:1] if len(shap_v.shape) > 1 else shap_v
        shap.summary_plot(s_plot_val, processed_df, plot_type="bar", show=False, color="#00f2ff", max_display=5)
        plt.xticks(color="#94a3b8", fontsize=8); plt.yticks(color="white", fontsize=8)
        # Fixed Matplotlib tuple color
        for spine in ax.spines.values(): spine.set_edgecolor((1, 1, 1, 0.1)) 
        st.pyplot(fig)
else:
    st.markdown("""<div style="text-align:center; padding:100px; opacity:0.5;">
        <p style="font-size:5rem;">🏎️</p>
        <p style="font-family:'Orbitron'; letter-spacing:3px;">SYSTEM STANDBY</p>
    </div>""", unsafe_allow_html=True)
