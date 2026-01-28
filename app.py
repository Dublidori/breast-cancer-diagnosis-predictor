"""
app.py -- Streamlit web app for breast cancer diagnosis prediction.

Run with: streamlit run app.py

Copyright (c) 2026 Miguel
Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International (CC BY-NC-SA 4.0). See https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import train_model, predict


# Page configuration (must be first Streamlit call)
st.set_page_config(
    page_title="Breast Cancer Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for responsive layout and polished UI
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        min-width: 380px;
        max-width: 420px;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        margin-bottom: 0.3rem;
    }

    /* Main content -- responsive */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 95%;
    }

    /* ---- Diagnosis card ---- */
    .dx-card {
        padding: 1.8rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .dx-card:hover { transform: translateY(-2px); }
    .dx-benign {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #4caf50;
        box-shadow: 0 4px 15px rgba(76,175,80,0.15);
    }
    .dx-malignant {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border: 2px solid #ef5350;
        box-shadow: 0 4px 15px rgba(239,83,80,0.15);
    }
    .dx-card .dx-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
        margin-bottom: 0.3rem;
    }
    .dx-card .dx-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .dx-benign .dx-value, .dx-benign .dx-label { color: #2e7d32; }
    .dx-malignant .dx-value, .dx-malignant .dx-label { color: #c62828; }

    /* ---- Confidence card ---- */
    .conf-card {
        padding: 1.8rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border: 2px solid #42a5f5;
        box-shadow: 0 4px 15px rgba(66,165,245,0.15);
        transition: transform 0.2s ease;
    }
    .conf-card:hover { transform: translateY(-2px); }
    .conf-card .conf-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #1565c0;
        opacity: 0.7;
        margin-bottom: 0.3rem;
    }
    .conf-card .conf-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1565c0;
        margin: 0;
    }
    .conf-bar-bg {
        width: 100%;
        height: 8px;
        background: rgba(21,101,192,0.15);
        border-radius: 4px;
        margin-top: 0.8rem;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #42a5f5, #1565c0);
        transition: width 0.5s ease;
    }

    /* ---- Stats row ---- */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin: 0.5rem 0 1.5rem 0;
        flex-wrap: wrap;
    }
    .stat-pill {
        flex: 1;
        min-width: 120px;
        padding: 0.7rem 1rem;
        border-radius: 10px;
        background: #f5f7fa;
        border-left: 4px solid #42a5f5;
    }
    .stat-pill .sp-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .stat-pill .sp-lbl {
        font-size: 0.72rem;
        color: #888;
        margin-top: 0.1rem;
    }

    /* ---- Test-tab result cards ---- */
    .test-card {
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .test-card:hover { transform: translateY(-1px); }
    .tc-benign {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 1.5px solid #4caf50;
        box-shadow: 0 2px 8px rgba(76,175,80,0.1);
    }
    .tc-malignant {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border: 1.5px solid #ef5350;
        box-shadow: 0 2px 8px rgba(239,83,80,0.1);
    }
    .tc-correct {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 1.5px solid #4caf50;
        box-shadow: 0 2px 8px rgba(76,175,80,0.1);
    }
    .tc-incorrect {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border: 1.5px solid #ff9800;
        box-shadow: 0 2px 8px rgba(255,152,0,0.1);
    }
    .test-card .tc-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.6;
        margin-bottom: 0.3rem;
    }
    .test-card .tc-value {
        font-size: 1.15rem;
        font-weight: 600;
    }

    /* ---- Misc ---- */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 24px; }
</style>
""", unsafe_allow_html=True)


# Cleaned-up feature labels (remove group prefix since expander already shows it)
CLEAN_NAMES = {
    "mean radius": "Radius",
    "mean texture": "Texture",
    "mean perimeter": "Perimeter",
    "mean area": "Area",
    "mean smoothness": "Smoothness",
    "mean compactness": "Compactness",
    "mean concavity": "Concavity",
    "mean concave points": "Concave Points",
    "mean symmetry": "Symmetry",
    "mean fractal dimension": "Fractal Dim.",
    "radius error": "Radius",
    "texture error": "Texture",
    "perimeter error": "Perimeter",
    "area error": "Area",
    "smoothness error": "Smoothness",
    "compactness error": "Compactness",
    "concavity error": "Concavity",
    "concave points error": "Concave Points",
    "symmetry error": "Symmetry",
    "fractal dimension error": "Fractal Dim.",
    "worst radius": "Radius",
    "worst texture": "Texture",
    "worst perimeter": "Perimeter",
    "worst area": "Area",
    "worst smoothness": "Smoothness",
    "worst compactness": "Compactness",
    "worst concavity": "Concavity",
    "worst concave points": "Concave Points",
    "worst symmetry": "Symmetry",
    "worst fractal dimension": "Fractal Dim.",
}


# Model loading (cached -- trains only once)
@st.cache_resource
def load_model():
    return train_model()


artifacts = load_model()

# Feature grouping: 10 mean, 10 SE, 10 worst
GROUPS = {
    "Mean Values": slice(0, 10),
    "Standard Error": slice(10, 20),
    "Worst Values": slice(20, 30),
}


# --- Resolve default values for sidebar inputs ---
# If a sample was loaded, use those values; otherwise use dataset means.
if "pending_sample" in st.session_state:
    defaults = st.session_state.pop("pending_sample")
else:
    defaults = [float(artifacts.feature_means[i]) for i in range(30)]


# --- Sidebar: Feature Inputs ---
st.sidebar.header("Cell Nucleus Measurements")
st.sidebar.caption("Adjust values or use **Load Sample** in the main area.")

input_values = np.zeros(30)

for group_name, idx_slice in GROUPS.items():
    with st.sidebar.expander(group_name, expanded=False):
        indices = range(30)[idx_slice]
        for i in indices:
            raw_name = artifacts.feature_names[i]
            label = CLEAN_NAMES.get(raw_name, raw_name)
            min_val = float(artifacts.feature_mins[i])
            max_val = float(artifacts.feature_maxs[i])

            padding = (max_val - min_val) * 0.1
            slider_min = max(0.0, min_val - padding)
            slider_max = max_val + padding

            input_values[i] = st.slider(
                label=label,
                min_value=slider_min,
                max_value=slider_max,
                value=defaults[i],
                format="%.4f",
                key=f"feature_{i}",
            )

st.sidebar.divider()
st.sidebar.caption(
    "**Disclaimer:** This is a prototype for educational purposes only. "
    "Not a medical diagnostic tool."
)


# --- Radar Chart Helper ---
def create_radar_chart(input_vals, arts):
    """Radar chart comparing current mean-feature input vs dataset averages."""
    labels = [CLEAN_NAMES.get(n, n) for n in arts.feature_names[:10]]
    N = len(labels)

    mins = arts.feature_mins[:10]
    maxs = arts.feature_maxs[:10]
    rng = maxs - mins
    rng[rng == 0] = 1

    inp = np.clip((input_vals[:10] - mins) / rng, 0, 1)
    avg = (arts.feature_means[:10] - mins) / rng

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    inp = np.concatenate([inp, [inp[0]]])
    avg = np.concatenate([avg, [avg[0]]])
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, inp, "o-", linewidth=2, color="#1976D2",
            label="Current Input", markersize=5)
    ax.fill(angles, inp, alpha=0.15, color="#1976D2")
    ax.plot(angles, avg, "o-", linewidth=2, color="#bdbdbd",
            label="Dataset Average", markersize=4)
    ax.fill(angles, avg, alpha=0.05, color="#9e9e9e")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Mean Feature Values (normalized)", size=12, pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# --- Main Area ---

st.title("Breast Cancer Diagnosis Predictor")

tab_predict, tab_test, tab_docs = st.tabs([
    "Prediction",
    "Test with Real Data",
    "How to Use",
])


# ===================== TAB 1: PREDICTION =====================
with tab_predict:

    # Stats row
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-pill">
            <div class="sp-val">{artifacts.accuracy:.1%}</div>
            <div class="sp-lbl">Model Accuracy</div>
        </div>
        <div class="stat-pill">
            <div class="sp-val">569</div>
            <div class="sp-lbl">Training Samples</div>
        </div>
        <div class="stat-pill">
            <div class="sp-val">30</div>
            <div class="sp-lbl">Input Features</div>
        </div>
        <div class="stat-pill">
            <div class="sp-val">100</div>
            <div class="sp-lbl">Decision Trees</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    label, confidence = predict(artifacts, input_values)

    # Diagnosis + Confidence cards
    col_dx, col_conf = st.columns(2, gap="large")

    with col_dx:
        card_cls = "dx-benign" if label == "Benign" else "dx-malignant"
        st.markdown(f"""
        <div class="dx-card {card_cls}">
            <div class="dx-label">Diagnosis</div>
            <p class="dx-value">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_conf:
        bar_pct = confidence * 100
        st.markdown(f"""
        <div class="conf-card">
            <div class="conf-label">Confidence</div>
            <p class="conf-value">{confidence:.1%}</p>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width: {bar_pct:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Radar chart + Feature Importance side by side
    col_radar, col_importance = st.columns(2, gap="medium")

    with col_radar:
        with st.expander("Feature Radar", expanded=True):
            st.caption(
                "Current input vs dataset average (mean values, normalized 0-1)"
            )
            fig_radar = create_radar_chart(input_values, artifacts)
            st.pyplot(fig_radar)
            plt.close(fig_radar)

    with col_importance:
        with st.expander("Feature Importance", expanded=True):
            st.caption(
                "Which features the Random Forest weighs most for classification"
            )

            importance_df = pd.DataFrame({
                "Feature": artifacts.feature_names,
                "Importance": artifacts.feature_importances,
            }).sort_values("Importance", ascending=True)

            fig_imp, ax_imp = plt.subplots(figsize=(8, 7))
            colors = [
                "#4CAF50" if imp > importance_df["Importance"].median()
                else "#90CAF9"
                for imp in importance_df["Importance"]
            ]
            ax_imp.barh(
                importance_df["Feature"],
                importance_df["Importance"],
                color=colors,
            )
            ax_imp.set_xlabel("Importance")
            ax_imp.set_title("Random Forest Feature Importances")
            ax_imp.spines["top"].set_visible(False)
            ax_imp.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.close(fig_imp)

    # Model Evaluation Details
    with st.expander("Model Evaluation Details", expanded=False):
        st.markdown("Classification report on the held-out test set:")
        st.code(artifacts.classification_report)


# ===================== TAB 2: TEST WITH REAL DATA =====================
with tab_test:
    st.markdown(
        "Load a real patient sample from the dataset to test the model. "
        "Compare the prediction against the known diagnosis."
    )

    st.markdown("")

    col_sample1, col_sample2 = st.columns([1, 1], gap="large")

    with col_sample1:
        sample_idx = st.number_input(
            "Sample index (0-568)",
            min_value=0,
            max_value=len(artifacts.dataset) - 1,
            value=0,
            step=1,
        )

    with col_sample2:
        filter_type = st.selectbox(
            "Filter by actual diagnosis",
            ["All", "Malignant only", "Benign only"],
        )

    if filter_type == "Malignant only":
        filtered = artifacts.dataset[artifacts.dataset["diagnosis"] == "Malignant"]
    elif filter_type == "Benign only":
        filtered = artifacts.dataset[artifacts.dataset["diagnosis"] == "Benign"]
    else:
        filtered = artifacts.dataset

    safe_idx = sample_idx % len(filtered)

    def load_sample():
        row = filtered.iloc[safe_idx]
        st.session_state["pending_sample"] = [
            float(row[fname]) for fname in artifacts.feature_names
        ]
        for i in range(30):
            st.session_state.pop(f"feature_{i}", None)

    st.button(
        "Load sample into sidebar inputs",
        type="primary",
        on_click=load_sample,
    )

    st.markdown("")

    # Show actual vs predicted for current sample
    sample_row = filtered.iloc[safe_idx]
    actual_diagnosis = sample_row["diagnosis"]
    sample_features = np.array([sample_row[f] for f in artifacts.feature_names])
    pred_label, pred_conf = predict(artifacts, sample_features)

    col_actual, col_pred, col_match = st.columns(3, gap="medium")

    with col_actual:
        cls = "tc-benign" if actual_diagnosis == "Benign" else "tc-malignant"
        st.markdown(f"""
        <div class="test-card {cls}">
            <div class="tc-label">Actual Diagnosis</div>
            <div class="tc-value">{actual_diagnosis}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_pred:
        cls = "tc-benign" if pred_label == "Benign" else "tc-malignant"
        st.markdown(f"""
        <div class="test-card {cls}">
            <div class="tc-label">Model Prediction</div>
            <div class="tc-value">{pred_label} ({pred_conf:.1%})</div>
        </div>
        """, unsafe_allow_html=True)

    with col_match:
        if pred_label == actual_diagnosis:
            cls, text = "tc-correct", "Correct"
        else:
            cls, text = "tc-incorrect", "Incorrect"
        st.markdown(f"""
        <div class="test-card {cls}">
            <div class="tc-label">Result</div>
            <div class="tc-value">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Dataset browser
    with st.expander("Browse dataset samples", expanded=False):
        display_df = filtered.reset_index(drop=True)
        st.dataframe(display_df.head(20), use_container_width=True)
        st.caption(f"Showing first 20 of {len(filtered)} samples.")


# ===================== TAB 3: HOW TO USE =====================
with tab_docs:

    # --- WHAT IS THIS APP ---
    st.markdown("### What is this app?")
    st.markdown("""
This app helps determine whether a breast tumor is **benign** (not cancer) or
**malignant** (cancer) based on measurements taken from a biopsy sample.

Here's what happens in real life:
1. A doctor performs a **Fine Needle Aspirate (FNA)** -- a thin needle is inserted
   into the tumor to extract a small tissue sample.
2. The sample is placed under a microscope, and a **digital image** is captured.
3. Software analyzes the image and measures the **shape, size, and texture** of
   the cell nuclei visible in the image.
4. Those measurements (30 numbers) are what this app uses to predict the diagnosis.

The model was trained on **569 real patient cases** from the University of Wisconsin
and gets the diagnosis right about **96% of the time**.
""")

    st.markdown("---")

    # --- HOW TO USE ---
    st.markdown("### How to use this app")

    st.markdown("#### Option A: Test with real patient data (easiest)")
    st.markdown("""
1. Click the **Test with Real Data** tab above.
2. Pick any **sample number** from 0 to 568 -- each one is a real patient.
3. You can filter to show only malignant or benign cases.
4. Click the blue **Load sample into sidebar inputs** button.
5. Go back to the **Prediction** tab -- you'll see the model's diagnosis
   and how confident it is.
6. The Test tab also shows whether the model got it right or wrong.
""")

    st.markdown("#### Option B: Enter values manually")
    st.markdown("""
1. Open the **sidebar** on the left (the panel with sliders).
2. Expand one of the three groups (Mean Values, Standard Error, Worst Values).
3. Move the sliders to set each measurement.
4. The **Prediction** tab updates live as you adjust values.
""")

    st.markdown("---")

    # --- THE THREE GROUPS ---
    st.markdown("### Understanding the three value groups")
    st.markdown("""
When the biopsy image is analyzed, the software looks at **all the cell nuclei**
in the sample (there can be dozens). For each measurement (like radius or area),
it doesn't just give one number -- it gives **three**:
""")

    st.success("""
**Mean Values** -- The **average** across all nuclei in the sample.

*Example: if 20 nuclei have radii of 10, 12, 11, 13... the mean radius
is the average of all of them. This tells you the "typical" nucleus size
in the sample.*
""")

    st.info("""
**Standard Error** -- How much the measurements **vary** from nucleus to nucleus.

*Example: if all nuclei have nearly the same radius, the standard error is low.
If some are tiny and others are huge, the standard error is high. High variation
can be a sign of abnormal cell growth.*
""")

    st.error("""
**Worst Values** -- The average of the **3 most extreme nuclei** (the largest,
most irregular ones).

*Example: out of all the nuclei, take the 3 with the biggest radius and average
them. This captures the "worst case" cells in the sample. Cancer cells tend to
be larger and more irregular, so high "worst" values are a strong signal of
malignancy.*
""")

    st.markdown("---")

    # --- WHAT EACH MEASUREMENT MEANS ---
    st.markdown("### What each measurement means")
    st.markdown("""
Each group contains the same 10 measurements. Here's what they actually measure
about the cell nuclei in the biopsy image:
""")

    st.markdown("""
| Measurement | Plain English explanation |
|---|---|
| **Radius** | How big the nucleus is -- the average distance from the center to the edge. Cancerous cells tend to be **larger**. |
| **Texture** | How uniform the grayscale shading is across the nucleus surface. Cancerous nuclei often have **uneven texture**. |
| **Perimeter** | The total length around the edge of the nucleus. Bigger and more irregular nuclei have a **longer perimeter**. |
| **Area** | The total surface area of the nucleus. Directly related to radius -- larger nuclei = larger area. |
| **Smoothness** | How smooth or bumpy the edge of the nucleus is. Cancerous nuclei often have **rougher, more irregular edges**. |
| **Compactness** | How round vs. elongated the nucleus is (calculated from perimeter and area). A perfect circle has low compactness. **Irregular shapes** score higher. |
| **Concavity** | How much the edge of the nucleus **caves inward**. Healthy nuclei are fairly round; cancerous ones often have dents and indentations. |
| **Concave Points** | How many separate **indentations** exist along the edge. More concave points = more irregular shape. This is one of the **most important** features for detection. |
| **Symmetry** | How symmetric the nucleus is -- would it look the same if you flipped it? Cancer cells are often **asymmetric**. |
| **Fractal Dimension** | How complex or "crinkly" the edge is (like measuring a coastline). Higher values mean a more **complex, irregular boundary**. |
""")

    st.markdown("---")

    # --- WHAT THE RESULTS MEAN ---
    st.markdown("### Understanding the results")
    st.markdown("""
When you go to the **Prediction** tab, you see two things:
""")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("""
**Diagnosis** -- Either:
- **Benign** (green) = the tumor is likely **not cancerous**. Benign tumors
  are non-threatening growths that don't spread.
- **Malignant** (red) = the tumor is likely **cancerous**. Malignant tumors
  can grow and spread to other parts of the body.
""")
    with col_r2:
        st.markdown("""
**Confidence** -- How sure the model is, from 0% to 100%.
- **Above 90%** = the model is quite confident.
- **70-90%** = moderate confidence, the case might be borderline.
- **Below 70%** = low confidence, the measurements are ambiguous.

The confidence comes from the Random Forest: 100 decision trees each "vote"
on the diagnosis, and the confidence is the percentage that agreed.
""")

    st.markdown("---")

    # --- KEY THINGS TO KNOW ---
    st.markdown("### Key things to know")
    st.markdown("""
- **This is NOT a medical tool.** It's a prototype for learning purposes.
  Real diagnosis requires pathologists, additional tests, and clinical context.
- **The model is ~96% accurate**, which means it gets about **4 out of 100
  cases wrong**. In medicine, that's not good enough on its own.
- **Default slider values are dataset averages**, which typically predict Benign.
  To see a Malignant prediction, load a malignant sample or increase the
  "Worst Values" sliders.
- **The most important features** for detection are usually:
  **Worst Concave Points**, **Worst Perimeter**, and **Mean Concave Points**.
  You can see the full ranking in the Feature Importance chart on the Prediction tab.
- The dataset comes from the **University of Wisconsin** (1995) and contains
  569 real cases: 357 benign and 212 malignant.
""")
