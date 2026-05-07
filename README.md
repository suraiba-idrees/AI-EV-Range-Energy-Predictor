# 🏎️ NEURA-DRIVE: AI-Powered EV Engine
> **Final Year Project:** An Advanced Electric Vehicle Range & Energy Consumption Predictor using Ensemble Learning.

![](https://shields.io)
![](https://shields.io)
![](https://shields.io)


## 🌟 Project Overview
**Neura-Drive** is an intelligent analytical dashboard designed to tackle one of the biggest challenges in the EV industry: **Range Anxiety**. By leveraging an ensemble of **Random Forest** and **XGBoost** models, it provides highly accurate predictions for vehicle range and energy consumption based on real-world driving conditions.

---

## 🧠 Key Features
*   **Real-time Telemetry Analysis:** Dynamically calculates impact based on speed, payload weight, and ambient weather conditions.
*   **Regenerative Braking Logic:** Simulates energy recovery efficiency and its direct impact on extending battery range.
*   **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** plots to provide transparency, showing exactly which factors influenced the AI's prediction.

---

## 🛠️ Tech Stack
*   **Language:** Python
*   **Web Framework:** Streamlit
*   **ML Frameworks:** Scikit-Learn, XGBoost
*   **Explainability:** SHAP
*   **Data Visualization:** Plotly, Matplotlib

---

## 📂 Project Structure
```text
├── data/               # Dataset files (CSV)
├── models/             # Trained model pickles (.pkl)
├── src/                # Source code and logic
└── requirements.txt    # Project dependencies
```

---

## 🚀 Installation & Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd AI-EV-Range-Energy-Predictor
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard:**
   ```bash
   streamlit run src/ev_prediction_dashboard.py
   ```
