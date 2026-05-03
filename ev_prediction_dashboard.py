import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import time

# --- 1. CONFIG & HIGH-END INTERACTIVE UI ---
st.set_page_config(page_title="NEURA-DRIVE | AI EV Engine", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=Rajdhani:wght@300;500;700&display=swap');

    .stApp {
        background: radial-gradient(circle at top left, rgba(0, 242, 255, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at bottom right, rgba(0, 242, 255, 0.05) 0%, transparent 30%),
                    #050505;
        background-attachment: fixed;
    }

    /* --- MOBILE VIEW FIXES (Sirf yeh 6 lines add ki hain) --- */
    @media (max-width: 768px) {
        .main-title { font-size: 1.8rem !important; gap: 10px !important; }
        .main-title img { width: 40px !important; }
        .val-number { font-size: 3.5rem !important; }
        .metric-card { padding: 20px !important; margin-bottom: 15px !important; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
    }

    /* 1. Sidebar Upgrade: Frosted Glass Panel Look */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(20, 20, 20, 0.98) 0%, rgba(10, 10, 10, 0.95) 100%) !important;
        border-right: 1px solid rgba(0, 242, 255, 0.2);
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding: 1.5rem;
        gap: 1rem;
    }

    /* 2. Interactive Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(25px) saturate(150%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 35px;
        border-radius: 30px;
        text-align: center; 
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .metric-card:hover {
        transform: scale(1.05) translateY(-8px);
        border: 1px solid rgba(0, 242, 255, 0.6);
    }

    /* 3. AI Box Spacing - PERFECTED BALANCE */
    .ai-box {
        background: rgba(0, 242, 255, 0.04);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        border-left: 6px solid #00f2ff;
        margin-top: 20px;
        margin-bottom: 12px;
        transition: all 0.4s ease;
    }
    
    .ai-box:hover {
        transform: scale(1.01);
        background: rgba(0, 242, 255, 0.07);
    }

    /* Expander Spacing - FIXED GAP */
    .stExpander {
        margin-top: 5px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.01) !important;
    }

    .main-title {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        letter-spacing: 6px;
        text-shadow: 0 0 25px rgba(0, 242, 255, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
    }

    .val-number { 
        color: #00f2ff; 
        font-size: 6.5rem; 
        font-weight: 900; 
        font-family: 'Rajdhani', sans-serif;
        line-height: 0.9;
    }
    
    .label-text { 
        color: #94a3b8; 
        font-size: 1rem; 
        letter-spacing: 3px; 
        margin-bottom: 8px;
        display: block;
    }

    .scan-bar {
        width: 100%;
        height: 2px;
        background: #00f2ff;
        box-shadow: 0 0 10px #00f2ff;
        position: relative;
        animation: scan 1.5s linear infinite;
    }

    @keyframes scan {
        0% { top: 0; }
        100% { top: 80px; opacity: 0; }
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

# --- 3. SIDEBAR ---
st.sidebar.markdown("<h2 style='color:#00f2ff; font-family:Orbitron; font-size:1.5rem;'>🕹️ MISSION CONTROL</h2>", unsafe_allow_html=True)
with st.sidebar.expander("🔋 Battery & Vehicle", expanded=True):
    capacity = st.number_input("Total Capacity (kWh)", 10, 150, 75)
    soc = st.slider("Current SOC (%)", 5, 100, 85)
    weight = st.number_input("Vehicle Weight (kg)", 1000, 3000, 1800)
with st.sidebar.expander("🛣️ Trip Settings", expanded=True):
    speed = st.slider("Target Speed (km/h)", 20, 160, 90)
    dist_planned = st.number_input("Trip Distance (km)", 1, 500, 50)
    mode = st.selectbox("Driving Mode", ["Normal", "Eco", "Sport"])
    style = st.radio("Driving Style", ["Conservative", "Aggressive"], horizontal=True)
with st.sidebar.expander("🌍 Environment", expanded=False):
    road = st.selectbox("Road", ["Highway", "City", "Rural"])
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Fog"])
    temp = st.slider("Temp (°C)", -15, 45, 25)
    hvac = st.toggle("Climate Control (AC/Heater)", value=True)

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

# --- 5. MAIN DASHBOARD ---
st.markdown("""
    <div class='main-title'>
        <img src='https://cdn-icons-png.flaticon.com/512/4712/4712035.png' width='80'>
        NEURA-DRIVE
    </div>
    """, unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; font-size:1.1rem; letter-spacing:5px; margin-bottom:40px;'>NEURAL ENERGY ANALYTICS ENGINE</p>", unsafe_allow_html=True)

if st.sidebar.button("🚀 EXECUTE MISSION", use_container_width=True):
    with st.empty():
        for i in range(3):
            st.markdown(f'<div class="ai-box"><div class="scan-bar"></div><p style="text-align:center; color:#00f2ff;">SCANNING NEURAL WEIGHTS... {30*(i+1)}%</p></div>', unsafe_allow_html=True)
            time.sleep(0.3)
        st.write("")

    range_km, energy, shap_v, processed_df = run_prediction()
    
    r_val = float(range_km[0]) if isinstance(range_km, (np.ndarray, list)) else float(range_km)
    e_val = float(energy[0]) if isinstance(energy, (np.ndarray, list)) else float(energy)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="metric-card">
                    <span class="label-text">PROJECTED RANGE</span>
                    <span class="val-number" style="text-shadow: 0 0 20px rgba(255, 255, 255, 0.28);">{r_val:.1f}</span><span style="color:#94a3b8; font-size:1.5rem;"> KM</span>
                    </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
                    <span class="label-text">TOTAL CONSUMPTION</span>
                    <span class="val-number" style="color:#ff4b4b; text-shadow: 0 0 25px rgba(255, 75, 75, 0.4);">{e_val:.2f}</span><span style="color:#94a3b8; font-size:1.5rem;"> kWh</span>
                    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True) 
    st.markdown("### 🧠 Diagnostic Report")
    shap_abs = np.abs(shap_v[0]) if len(shap_v.shape) > 1 else np.abs(shap_v)
    
    # --- FIXED THE ERROR ON THE LINE BELOW ---
    top_feature = model_columns[np.argmax(shap_abs)].replace('_', ' ') if hasattr(model_columns[np.argmax(shap_abs)], 'replace') else str(model_columns[np.argmax(shap_abs)])
    
    st.markdown(f"""
    <div class="ai-box">
        <p style="color:#00f2ff; font-weight:bold; font-size:1.2rem; font-family:'Orbitron';">🔍 RATIONALE</p>
        <p style="color:#cbd5e1;">AI has identified <b>{top_feature}</b> as the primary influencer. Mission reach is optimized for current <b>{road}</b> conditions.</p>
        <p style="color:#10b981; font-weight:bold; font-size:1.2rem; margin-top:15px; font-family:'Orbitron';">💡 STRATEGY</p>
        <p style="color:#cbd5e1;">A 15% reduction in speed could extend your mission to <b>{r_val*1.15:.1f} KM</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Technical Feature Analysis (XAI)", expanded=False):
        fig, ax = plt.subplots(figsize=(7, 2.5))
        fig.patch.set_alpha(0); ax.set_facecolor('none')
        s_plot_val = shap_v[0:1] if len(shap_v.shape) > 1 else shap_v
        shap.summary_plot(s_plot_val, processed_df, plot_type="bar", show=False, color="#00f2ff", max_display=5)
        plt.xticks(color="#94a3b8", fontsize=8); plt.yticks(color="white", fontsize=8)
        st.pyplot(fig)
else:
    st.info("System Ready. Please configure mission parameters in the sidebar to begin.")
