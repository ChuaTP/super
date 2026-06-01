import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Heart Failure Mortality Analyzer",
    layout="wide"
)

# ====================== RED ECG HEARTBEAT BACKGROUND ======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Roboto+Mono&display=swap');

    .stApp {
        background-color: #0a0000;
        color: #ffffff;
    }
    
    label, div[data-testid="stWidgetLabel"] p {
        color: #f0f0f0 !important; /* Bright white/gray */
        font-weight: 500 !important;
        letter-spacing: 0.5px;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            linear-gradient(to right, transparent 40%, rgba(255, 30, 30, 0.15) 50%, transparent 60%),
            repeating-linear-gradient(
                90deg,
                transparent,
                transparent 80px,
                #ff1a1a 82px,
                #ff1a1a 85px,
                transparent 87px
            );
        background-size: 300% 100%;
        animation: ecg-move 8s linear infinite;
        pointer-events: none;
        z-index: -1;
        opacity: 0.75;
    }

    @keyframes ecg-move {
        0% { background-position: 0 0; }
        100% { background-position: -300% 0; }
    }

    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(255, 50, 50, 0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 50, 50, 0.08) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: -1;
        opacity: 0.4;
    }

    .main-header {
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        font-size: 3.1rem;
        color: #ff3333;
        text-shadow: 0 0 25px #ff0000;
        margin-bottom: 8px;
    }

    .sub-header {
        font-family: 'Roboto Mono', monospace;
        text-align: center;
        color: #ff8888;
        letter-spacing: 5px;
        font-size: 1.15rem;
    }

    .stButton>button {
        background: linear-gradient(45deg, #ff3333, #aa0000);
        color: white;
        border: 2px solid #ff5555;
        font-weight: bold;
        font-family: 'Orbitron', sans-serif;
        padding: 12px;
    }

    .stButton>button:hover {
        box-shadow: 0 0 30px #ff3333;
        transform: scale(1.04);
    }
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown('<h1 class="main-header"> HEART FAILURE RISK ANALYZER</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">CONTINUOUS MORTALITY MONITORING SYSTEM</p>', unsafe_allow_html=True)
st.markdown("---")

# ====================== LOAD MODEL + SCALER ======================
@st.cache_resource(show_spinner="Loading Diagnostic Model...")
def load_model_and_scaler():
    model_path = Path("lgbm_model.pkl")

    if not model_path.exists():
        st.error("❌ lgbm_model.pkl not found! Please run lightgbm_model.py first.")
        st.stop()

    # ── Load the combined dictionary saved by lightgbm_model.py ──
    model_data = joblib.load(model_path)
    model      = model_data["model"]
    scaler     = model_data["scaler"]

    return model, scaler

model, scaler = load_model_and_scaler()
st.success("Model Loaded | System Ready")

# ====================== PATIENT INPUT ======================
st.header("🫀 Enter Patient Vital Signs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=60, step=1)
    anaemia = st.selectbox("Anaemia", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, value=500, step=10)
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=38, step=1)
    high_blood_pressure = st.selectbox("High Blood Pressure", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0, value=260000, step=1000)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.2, format="%.2f")
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=0, value=136, step=1)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

# ====================== PREDICTION ======================
if st.button(" RUN RISK ANALYSIS", type="primary", use_container_width=True):

    input_data = pd.DataFrame([{
        "age"                     : age,
        "anaemia"                 : anaemia,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "ejection_fraction"       : ejection_fraction,
        "high_blood_pressure"     : high_blood_pressure,
        "platelets"               : platelets,
        "serum_creatinine"        : serum_creatinine,
        "serum_sodium"            : serum_sodium,
        "sex"                     : sex
    }])

    input_scaled = scaler.transform(input_data)
    probability  = model.predict_proba(input_scaled)[0][1]

    st.markdown("###  ANALYSIS COMPLETE")

    if probability > 0.5:
        st.error(" HIGH RISK — IMMEDIATE MEDICAL INTERVENTION RECOMMENDED")
        risk_text  = "HIGH RISK"
        risk_color = "🔴"
    else:
        st.success(" LOW RISK — PATIENT CONDITION APPEARS STABLE")
        risk_text  = "LOW RISK"
        risk_color = "🟢"

    st.metric(label="**RISK ASSESSMENT**", value=f"{risk_color} {risk_text}")


    st.caption("Early warning system for heart failure mortality risk")

    with st.expander(" Raw Patient Data"):
        st.dataframe(input_data, use_container_width=True)

# ====================== FOOTER ======================
st.markdown("---")
st.caption("RSW2S2G3 • Lead: KOH HUAI YU | CHUA THIAM POH | CHEN JENG JUN")