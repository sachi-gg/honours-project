"""
app.py
=============================================================================
Streamlit Clinical Web Interface for Anemia Detection
- Uses CBC Blood Test features: Gender, Hemoglobin, MCH, MCHC, MCV
- Preprocessing: StandardScaler
- Model: Trained RandomForestClassifier
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Anemia Risk Detector | CBC Analysis",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern medical styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.8rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-normal {
        background-color: #dcfce7;
        color: #15803d;
    }
    .badge-risk {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load pre-trained model and standard scaler from disk."""
    model_path = "anemia_model.pkl"
    scaler_path = "scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ---------------------------------------------------------
# Sidebar: Clinical References & Project Details
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/drop-of-blood.png",
        width=64
    )
    st.title("Clinical Reference")
    st.markdown("Standard Complete Blood Count (CBC) adult reference ranges:")

    ref_data = {
        "Metric": ["Hemoglobin (Male)", "Hemoglobin (Female)", "MCV", "MCH", "MCHC"],
        "Normal Range": ["13.5 - 17.5 g/dL", "12.0 - 15.5 g/dL", "80.0 - 100.0 fL", "27.0 - 33.0 pg", "32.0 - 36.0 g/dL"]
    }
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

    with st.expander("ℹ️ About CBC Markers"):
        st.markdown("""
        - **Hemoglobin (Hb):** Primary oxygen-carrying protein in red blood cells.
        - **MCV:** Average volume/size of red blood cells. Low indicates microcytosis.
        - **MCH:** Average mass of hemoglobin per red cell.
        - **MCHC:** Concentration of hemoglobin in a given volume of packed red cells.
        """)

    st.divider()
    st.markdown("**Model:** Random Forest Classifier")
    st.markdown("**Dataset:** Anemia CBC Dataset")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">🩸 Anemia Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Automated screening tool powered by Machine Learning and Complete Blood Count (CBC) parameters.</div>',
    unsafe_allow_html=True
)

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error("⚠️ Model artifacts not found (`anemia_model.pkl` or `scaler.pkl`).")
    st.info("Please train the model first by running: `python train.py` in your terminal.")
    st.stop()


# ---------------------------------------------------------
# Input Controls (2-Column Clinical Layout)
# ---------------------------------------------------------
st.markdown("### Patient Clinical Parameters")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Demographics & Core Marker")
    
    # 1. Gender input (Male: 1, Female: 0)
    gender_str = st.radio(
        "Biological Gender",
        options=["Female", "Male"],
        index=1,
        help="Biological sex determines normal physiological hemoglobin thresholds.",
        horizontal=True
    )
    gender_val = 1 if gender_str == "Male" else 0

    # 2. Hemoglobin input
    hemoglobin = st.slider(
        "Hemoglobin (Hb)",
        min_value=4.0,
        max_value=20.0,
        value=13.5,
        step=0.1,
        format="%.1f g/dL",
        help="Standard adult range: 12.0 - 17.5 g/dL depending on gender."
    )

with col2:
    st.markdown("#### Red Cell Indices")
    
    # 3. MCV input
    mcv = st.slider(
        "Mean Corpuscular Volume (MCV)",
        min_value=50.0,
        max_value=120.0,
        value=88.0,
        step=0.5,
        format="%.1f fL",
        help="Average volume of a red blood cell. Normal: 80 - 100 fL."
    )

    # 4. MCH input
    mch = st.slider(
        "Mean Corpuscular Hemoglobin (MCH)",
        min_value=15.0,
        max_value=45.0,
        value=30.0,
        step=0.5,
        format="%.1f pg",
        help="Average amount of hemoglobin per red cell. Normal: 27 - 33 pg."
    )

    # 5. MCHC input
    mchc = st.slider(
        "Mean Corpuscular Hemoglobin Concentration (MCHC)",
        min_value=25.0,
        max_value=40.0,
        value=33.5,
        step=0.2,
        format="%.1f g/dL",
        help="Average concentration of hemoglobin inside a red cell. Normal: 32 - 36 g/dL."
    )

st.write("") # Spacing

# ---------------------------------------------------------
# Prediction & Diagnostic Assessment
# ---------------------------------------------------------
diagnose_clicked = st.button("🔬 Diagnose Risk", type="primary", use_container_width=True)

if diagnose_clicked:
    # Prepare input array matching training order: Gender, Hemoglobin, MCH, MCHC, MCV
    input_df = pd.DataFrame(
        [[gender_val, hemoglobin, mch, mchc, mcv]],
        columns=["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
    )

    # Apply standard scaling
    input_scaled = scaler.transform(input_df)

    # Predict class & confidence
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence_pct = probabilities[prediction] * 100

    st.markdown("---")
    st.subheader("Diagnostic Assessment Result")

    # Result Banner
    if prediction == 1:
        st.error("### ⚠️ High Risk of Anemia detected")
        status_msg = "The patient's CBC profile matches clinical indicators associated with Anemia."
    else:
        st.success("### ✅ Normal / Non-Anemic profile")
        status_msg = "The patient's blood test parameters fall within expected non-anemic ranges."

    st.write(status_msg)

    # Metric Cards: Confidence & Marker Analysis
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.metric(
            label="Prediction Confidence",
            value=f"{confidence_pct:.1f}%"
        )
        st.progress(float(confidence_pct / 100))

    with res_col2:
        hb_normal_min = 13.5 if gender_val == 1 else 12.0
        hb_normal_max = 17.5 if gender_val == 1 else 15.5
        
        if hemoglobin < hb_normal_min:
            hb_status = "Below Normal (Low)"
        elif hemoglobin > hb_normal_max:
            hb_status = "Above Normal (High)"
        else:
            hb_status = "Within Normal Range"
            
        st.metric(
            label="Hemoglobin Level",
            value=f"{hemoglobin:.1f} g/dL",
            delta=hb_status,
            delta_color="normal" if hb_status == "Within Normal Range" else "inverse"
        )

    with res_col3:
        cell_size = "Microcytic (< 80 fL)" if mcv < 80 else ("Macrocytic (> 100 fL)" if mcv > 100 else "Normocytic (80-100 fL)")
        st.metric(
            label="Red Cell Morphology (MCV)",
            value=cell_size
        )

    # Hemoglobin Visual Indicator Relative to Normal Range
    st.markdown("#### Hemoglobin Position Relative to Clinical Norms")
    
    # Calculate visual normalized gauge percentage (clamped between 4.0 and 20.0 g/dL)
    min_scale, max_scale = 4.0, 20.0
    clamped_hb = max(min_scale, min(max_scale, hemoglobin))
    gauge_percent = ((clamped_hb - min_scale) / (max_scale - min_scale)) * 100

    normal_min_pct = ((hb_normal_min - min_scale) / (max_scale - min_scale)) * 100
    normal_max_pct = ((hb_normal_max - min_scale) / (max_scale - min_scale)) * 100

    st.markdown(f"""
    <div style="background-color: #f1f5f9; border-radius: 10px; padding: 16px; margin-top: 10px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #64748b; margin-bottom: 6px;">
            <span>Severe Deficiency (4.0 g/dL)</span>
            <span>Expected Normal ({hb_normal_min:.1f} - {hb_normal_max:.1f} g/dL for {gender_str})</span>
            <span>High (20.0 g/dL)</span>
        </div>
        <div style="position: relative; height: 24px; background: #e2e8f0; border-radius: 12px; overflow: hidden;">
            <!-- Normal healthy zone indicator -->
            <div style="position: absolute; left: {normal_min_pct:.1f}%; width: {(normal_max_pct - normal_min_pct):.1f}%; height: 100%; background: #bbf7d0; opacity: 0.9;"></div>
            <!-- Patient value indicator -->
            <div style="position: absolute; left: 0%; width: {gauge_percent:.1f}%; height: 100%; background: {'#ef4444' if hemoglobin < hb_normal_min else '#22c55e'}; border-radius: 12px; transition: width 0.3s ease;"></div>
        </div>
        <div style="margin-top: 8px; font-size: 0.9rem; font-weight: 600; color: #334155;">
            Current Hemoglobin: <span style="color: {'#dc2626' if hemoglobin < hb_normal_min else '#16a34a'};">{hemoglobin:.1f} g/dL</span> — Status: <em>{hb_status}</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer / Academic Disclaimer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** Academic prototype for screening demonstration only; "
    "not a substitute for professional clinical diagnosis or medical consultation."
)
