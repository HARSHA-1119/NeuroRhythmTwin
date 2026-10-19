import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prediction_model import (
    predict_demo_event,
    get_demo_model_metrics,
)
from pathlib import Path

st.set_page_config(
    page_title="NeuroRhythm Twin",
    page_icon="🧠",
    layout="wide"
)


# Futuristic NeuroRhythm Twin Theme
st.markdown("""
<style>
/* Import futuristic typography */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* Main application background */
.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(0, 115, 255, 0.15), transparent 38%),
        radial-gradient(ellipse at 90% 25%, rgba(0, 238, 255, 0.08), transparent 35%),
        linear-gradient(145deg, #020817 0%, #06152b 50%, #020817 100%);
    color: #e6f7ff;
    font-family: 'Inter', sans-serif;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1800px;
}

/* Main headings */
h1, h2, h3 {
    color: #eafaff !important;
    letter-spacing: 0.4px;
}

h1 {
    font-family: 'Orbitron', sans-serif !important;
    text-shadow: 0 0 18px rgba(0, 210, 255, 0.4);
}

h2, h3 {
    text-shadow: 0 0 12px rgba(0, 180, 255, 0.16);
}

/* Paragraphs and labels */
p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: #b8cee4;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06152b 0%, #030b19 100%);
    border-right: 1px solid rgba(0, 217, 255, 0.35);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #49e8ff !important;
}

/* Glass-style metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(10, 38, 69, 0.92),
        rgba(4, 18, 38, 0.92)
    );
    border: 1px solid rgba(0, 205, 255, 0.42);
    border-radius: 15px;
    padding: 17px 18px;
    box-shadow:
        0 0 15px rgba(0, 174, 255, 0.08),
        inset 0 0 18px rgba(0, 174, 255, 0.035);
    transition: all 0.25s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #32e6ff;
    box-shadow:
        0 0 22px rgba(0, 205, 255, 0.22),
        inset 0 0 18px rgba(0, 174, 255, 0.07);
    transform: translateY(-2px);
}

[data-testid="stMetricLabel"] {
    color: #91c8e8 !important;
}

[data-testid="stMetricValue"] {
    color: #f1fcff !important;
    text-shadow: 0 0 12px rgba(0, 220, 255, 0.28);
}

/* Bordered containers and expandable panels */
[data-testid="stExpander"],
[data-testid="stForm"] {
    background: rgba(5, 23, 45, 0.78);
    border: 1px solid rgba(0, 190, 255, 0.28);
    border-radius: 14px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    color: #e9fcff;
    background: linear-gradient(110deg, #0757a8, #087eac);
    border: 1px solid #28dfff;
    border-radius: 10px;
    transition: all 0.25s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    color: #ffffff;
    border-color: #a4f5ff;
    box-shadow: 0 0 18px rgba(0, 205, 255, 0.4);
    transform: translateY(-1px);
}

/* Inputs */
.stTextInput input,
.stSelectbox [data-baseweb="select"],
.stNumberInput input {
    background-color: #081c33;
    color: #e9fcff;
    border-color: rgba(0, 200, 255, 0.45);
    border-radius: 8px;
}

/* Alerts */
[data-testid="stAlert"] {
    background: rgba(7, 34, 59, 0.85);
    border-radius: 12px;
    border: 1px solid rgba(0, 190, 255, 0.28);
}

/* Chart containers */
[data-testid="stPlotlyChart"] {
    background: rgba(4, 18, 36, 0.72);
    border: 1px solid rgba(0, 190, 255, 0.23);
    border-radius: 14px;
    padding: 8px;
}

/* Dividers */
hr {
    border-color: rgba(0, 190, 255, 0.25) !important;
}

/* Subtle entrance animation */
@keyframes neonFadeIn {
    from {
        opacity: 0;
        transform: translateY(7px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

[data-testid="stVerticalBlock"] {
    animation: neonFadeIn 0.45s ease-out;
}

/* Respect reduced-motion preferences */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

/* Mobile spacing */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.7rem 2rem;
    }

    h1 {
        font-size: 1.65rem !important;
    }
}
.neuro-panel {
    background: linear-gradient(
        120deg,
        rgba(8, 28, 52, 0.94),
        rgba(13, 18, 48, 0.94)
    );
    border: 1px solid rgba(53, 223, 255, 0.35);
    border-radius: 18px;
    padding: 22px;
    margin: 18px 0 26px 0;
    box-shadow:
        0 0 22px rgba(0, 195, 255, 0.08),
        inset 0 0 22px rgba(0, 195, 255, 0.035);
}

.neuro-panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    color: #55eaff;
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    letter-spacing: 1.5px;
    padding-bottom: 15px;
    border-bottom: 1px solid rgba(53, 223, 255, 0.18);
}

.neuro-status {
    color: #75ffc5;
    font-size: 10px;
}

.neuro-status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: #75ffc5;
    border-radius: 50%;
    margin-right: 7px;
    box-shadow: 0 0 10px #75ffc5;
    animation: neuroPulse 1.8s infinite;
}

.neuro-panel-body {
    display: flex;
    align-items: center;
    gap: 25px;
    padding-top: 22px;
}

.neuro-orb {
    width: 92px;
    height: 92px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    border: 1px solid #35dfff;
    background: radial-gradient(
        circle,
        rgba(53, 223, 255, 0.2),
        rgba(65, 70, 255, 0.08) 55%,
        transparent 72%
    );
    box-shadow:
        0 0 18px rgba(53, 223, 255, 0.35),
        inset 0 0 18px rgba(53, 223, 255, 0.2);
    animation: neuroFloat 3s ease-in-out infinite;
}

.neuro-orb-inner {
    color: #ffffff;
    font-family: 'Orbitron', sans-serif;
    font-size: 26px;
    font-weight: 800;
    text-shadow: 0 0 14px #35dfff;
}

.neuro-system-text {
    flex: 1;
    min-width: 0;
}

.neuro-system-text h3 {
    color: #e5f7ff;
    font-size: 20px;
    margin: 0 0 8px 0;
}

.neuro-system-text p {
    color: #9dbbd4;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 14px;
}

.neuro-progress {
    height: 4px;
    width: 100%;
    overflow: hidden;
    border-radius: 8px;
    background: rgba(100, 160, 200, 0.15);
}

.neuro-progress-fill {
    height: 100%;
    width: 72%;
    background: linear-gradient(90deg, #35dfff, #8174ff);
    box-shadow: 0 0 12px rgba(53, 223, 255, 0.8);
    border-radius: 8px;
}

.neuro-system-text small {
    display: block;
    color: #6f9bb8;
    font-size: 9px;
    letter-spacing: 1.2px;
    margin-top: 10px;
}

@keyframes neuroPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

@keyframes neuroFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}

@media (max-width: 600px) {
    .neuro-panel-body {
        align-items: flex-start;
        gap: 15px;
    }

    .neuro-orb {
        width: 65px;
        height: 65px;
    }

    .neuro-orb-inner {
        font-size: 20px;
    }

    .neuro-system-text h3 {
        font-size: 16px;
    }
}

@media (prefers-reduced-motion: reduce) {
    .neuro-status-dot,
    .neuro-orb {
        animation: none;
    }
}
/* Futuristic dashboard section headings */
h2, h3 {
    color: #DDF7FF;
    letter-spacing: 0.5px;
}

div[data-testid="stSubheader"] {
    color: #45E5FF;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(
        90deg,
        transparent,
        #35DFFF,
        #8174FF,
        transparent
    );
    opacity: 0.7;
    margin: 26px 0;
}

.neuro-section-label {
    font-family: 'Orbitron', sans-serif;
    color: #45E5FF;
    letter-spacing: 2px;
    font-size: 12px;
    text-transform: uppercase;
    text-shadow: 0 0 10px rgba(53, 223, 255, 0.4);
}

.neuro-wave-panel {
    margin: 20px 0;
    padding: 20px;
    border: 1px solid rgba(53, 223, 255, 0.3);
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        rgba(5, 20, 43, 0.95),
        rgba(20, 15, 52, 0.95)
    );
    box-shadow: inset 0 0 24px rgba(53, 223, 255, 0.04);
}

.neuro-wave-heading,
.neuro-wave-footer {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    font-family: 'Orbitron', sans-serif;
    font-size: 10px;
    letter-spacing: 1.4px;
}

.neuro-wave-heading {
    color: #55eaff;
}

.neuro-wave-tag {
    color: #c1a5ff;
}

.neuro-wave {
    position: relative;
    height: 125px;
    overflow: hidden;
    margin: 18px 0;
    border-radius: 12px;
    background:
        linear-gradient(rgba(53,223,255,.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(53,223,255,.05) 1px, transparent 1px);
    background-size: 22px 22px;
}

.neuro-wave-line {
    position: absolute;
    top: 50%;
    left: -10%;
    width: 120%;
    height: 3px;
    background: #35dfff;
    border-radius: 50%;
    box-shadow: 0 0 8px #35dfff, 0 0 20px #35dfff;
    transform: skewY(-7deg);
    animation: neuroWaveMove 3s ease-in-out infinite alternate;
}

.neuro-wave-line-two {
    top: 62%;
    height: 2px;
    background: #9677ff;
    box-shadow: 0 0 8px #9677ff, 0 0 18px #9677ff;
    transform: skewY(5deg);
    animation-duration: 4s;
    animation-direction: alternate-reverse;
}

.neuro-wave-footer {
    color: #83a7c4;
    font-size: 9px;
}

@keyframes neuroWaveMove {
    from {
        transform: translateY(-12px) skewY(-7deg) scaleY(1);
    }
    to {
        transform: translateY(12px) skewY(7deg) scaleY(1.7);
    }
}

@media (prefers-reduced-motion: reduce) {
    .neuro-wave-line {
        animation: none;
    }
}

</style>
""", unsafe_allow_html=True)


# Futuristic Dashboard Header
header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        """
        <div style="padding: 8px 0 18px 0;">
            <div style="
                color: #35e5ff;
                font-size: 11px;
                letter-spacing: 4px;
                font-weight: 700;
                margin-bottom: 10px;
            ">
                NEURAL SYSTEMS / DIGITAL HEALTH PLATFORM
            </div>
            <h1 style="
                font-family: Orbitron, sans-serif;
                font-size: clamp(26px, 3vw, 42px);
                margin: 0;
                color: #e9fbff;
                text-shadow: 0 0 22px rgba(0, 210, 255, 0.5);
            ">
                NEURORHYTHM <span style="color:#32dfff;">TWIN</span>
            </h1>
            <div style="
                color: #9bbbd7;
                font-size: 14px;
                margin-top: 9px;
            ">
                Understand Today. Build a Healthier Tomorrow.
            </div>
            <div style="
                width: 190px;
                height: 2px;
                margin-top: 17px;
                background: linear-gradient(90deg, #25e6ff, #5d68ff, transparent);
                box-shadow: 0 0 12px rgba(37, 230, 255, 0.65);
            "></div>
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        """
        <div style="
            margin-top: 20px;
            padding: 13px 15px;
            background: linear-gradient(135deg, rgba(8,42,65,.9), rgba(4,19,39,.9));
            border: 1px solid rgba(45,225,255,.45);
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 0 18px rgba(0,180,255,.08);
        ">
            <div style="color:#49f5c8;font-size:12px;letter-spacing:1px;">
                ● SYSTEM ONLINE
            </div>
            <div style="color:#91b9d8;font-size:10px;margin-top:6px;">
                SYNTHETIC DEMO MODE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.caption(
    "Research prototype · Historical sample data and simulated current signals"
)

st.info(
    "Research demonstration only. Current signals and prediction labels "
    "are synthetic. This prototype is not clinically validated and "
    "must not be used for diagnosis or treatment decisions."
)
st.subheader("Your Personal Digital Health Twin")

st.info(
    "Prototype mode: all health readings below are simulated. "
    "They are not real patient measurements or medical advice."
)

# Sidebar controls
st.sidebar.header("Twin Controls")

profile_name = st.sidebar.text_input(
    "Demo profile name",
    value="Demo Participant",
    key="demo_profile_name"
)

activity_context = st.sidebar.selectbox(
    "Current activity context",
    ["Studying", "Working", "Relaxing", "Exercising", "Sleeping"]
)


st.sidebar.caption(
    "Use fictional information only. No real health records are stored."
)
hours = st.sidebar.slider(
    "Hours of simulated data",
    min_value=6,
    max_value=48,
    value=24,
    step=6,
    key="hours_of_simulated_data"
)

# Generate example wearable data
np.random.seed(42)
timestamps = pd.date_range(
    end=pd.Timestamp.now(),
    periods=hours,
    freq="h"
)

health_data = pd.DataFrame({
    "Time": timestamps,
    "Heart Rate (bpm)": np.random.normal(75, 8, hours).clip(50, 120).round(),
    "HRV (ms)": np.random.normal(45, 10, hours).clip(15, 100).round(),
    "Sleep (hours)": np.random.normal(7, 0.8, hours).clip(3, 10).round(1),
    "Stress Index (simulated)": np.random.randint(20, 81, hours)
})

# Latest example readings
latest = health_data.iloc[-1]

st.write(f"**Demo profile:** {profile_name}")
st.write(f"**Activity context:** {activity_context}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Heart Rate", f"{latest['Heart Rate (bpm)']:.0f} bpm")
col2.metric("HRV", f"{latest['HRV (ms)']:.0f} ms")
col3.metric("Sleep", f"{latest['Sleep (hours)']:.1f} h")
col4.metric("Stress Index", f"{latest['Stress Index (simulated)']}/100")


# Futuristic Health Signal Overview
st.markdown("## ◈ LIVE SIGNAL OVERVIEW")
st.caption("CURRENT SIMULATED READINGS · DEMONSTRATION DATA")

heart_rate_card = float(latest["Heart Rate (bpm)"])
hrv_card = float(latest["HRV (ms)"])
stress_card = float(latest["Stress Index (simulated)"])

# Use the actual sleep column if it exists in the latest row.
sleep_candidates = [
    col for col in latest.index
    if "sleep" in str(col).lower()
]

card1, card2, card3, card4 = st.columns(4)

with card1:
    st.metric(
        "♡ Heart Rate",
        f"{heart_rate_card:.1f} bpm"
    )

with card2:
    st.metric(
        "∿ HRV",
        f"{hrv_card:.1f} ms"
    )

with card3:
    st.metric(
        "✧ Simulated Stress",
        f"{stress_card:.1f}"
    )

with card4:
    if sleep_candidates:
        sleep_value = float(latest[sleep_candidates[0]])
        st.metric("☾ Sleep Metric", f"{sleep_value:.1f}")
    else:
        st.metric("☾ Sleep Metric", "N/A")

st.caption(
    "All values are from the demonstration dataset. "
    "The stress index is simulated, and these cards are not a medical assessment."
)

# Animated Neural Waveform
st.markdown(
    """
    <div class="neuro-wave-panel">
        <div class="neuro-wave-heading">
            <span>◈ NEURAL SIGNAL VISUALIZER</span>
            <span class="neuro-wave-tag">SIMULATION</span>
        </div>
        <div class="neuro-wave">
            <div class="neuro-wave-line"></div>
            <div class="neuro-wave-line neuro-wave-line-two"></div>
        </div>
        <div class="neuro-wave-footer">
            <span>SIMULATED SIGNAL</span>
            <span>VISUAL DEMONSTRATION ONLY</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="neuro-section-label">
        ◈ PREDICTIVE INTELLIGENCE MODULE
    </div>
    """,
    unsafe_allow_html=True
)

# Signal values linked to the simulated dataset
st.markdown("#### ◈ SIGNAL TELEMETRY")

telemetry1, telemetry2, telemetry3 = st.columns(3)

with telemetry1:
    st.metric(
        "Heart Rate",
        f"{float(latest['Heart Rate (bpm)']):.1f} bpm"
    )

with telemetry2:
    st.metric(
        "Heart Rate Variability",
        f"{float(latest['HRV (ms)']):.1f} ms"
    )

with telemetry3:
    st.metric(
        "Simulated Stress Index",
        f"{float(latest['Stress Index (simulated)']):.1f}"
    )

st.caption(
    "Telemetry reflects the latest row in the synthetic demonstration dataset. "
    "It is not a real-time wearable connection or clinical assessment."
)

st.markdown(
    "## AI Pattern Analysis"
)

st.caption(
    "Exploratory model output based on synthetic demonstration signals"
)

st.divider()

# AI Prediction Demo
st.divider()
st.header("🤖 AI Signal Prediction Demo")

st.write(
    "This experimental model estimates a synthetic event label from "
    "simulated signals. It is not a medical risk score or a validated forecast."
)

prediction_score = predict_demo_event(
    heart_rate=float(latest["Heart Rate (bpm)"]),
    hrv=float(latest["HRV (ms)"]),
    stress_index=float(latest["Stress Index (simulated)"])
)


# AI Model Evaluation - Synthetic Demonstration
st.divider()
st.markdown("### ◈ AI MODEL EVALUATION")

demo_metrics = get_demo_model_metrics()
positive_metrics = demo_metrics["report"]["Elevated demo pattern"]

eval_col1, eval_col2, eval_col3 = st.columns(3)

with eval_col1:
    st.metric(
        "Test Accuracy",
        f"{demo_metrics['accuracy'] * 100:.1f}%"
    )

with eval_col2:
    st.metric(
        "Precision",
        f"{positive_metrics['precision'] * 100:.1f}%"
    )

with eval_col3:
    st.metric(
        "Recall",
        f"{positive_metrics['recall'] * 100:.1f}%"
    )

st.metric(
    "F1-Score",
    f"{positive_metrics['f1-score'] * 100:.1f}%"
)

st.caption(
    f"Evaluation used {demo_metrics['test_samples']} held-out "
    "synthetic samples. These metrics measure agreement with the "
    "synthetic labeling rule, not clinical accuracy or real-world "
    "health prediction performance."
)



# Signal Snapshot - Demonstration Data
st.markdown("### ◈ SIGNAL SNAPSHOT")

snapshot_col1, snapshot_col2, snapshot_col3 = st.columns(3)

with snapshot_col1:
    st.metric(
        "Heart Rate",
        f"{float(latest['Heart Rate (bpm)']):.1f} bpm"
    )

with snapshot_col2:
    st.metric(
        "HRV",
        f"{float(latest['HRV (ms)']):.1f} ms"
    )

with snapshot_col3:
    st.metric(
        "Simulated Stress",
        f"{float(latest['Stress Index (simulated)']):.1f}"
    )

st.caption(
    "Snapshot of the latest synthetic demonstration values. "
    "These measurements are not a medical assessment."
)


# AI Insight Summary - Experimental Demo
st.markdown(
    """
    <div style="
        padding:18px;
        margin:16px 0;
        border:1px solid rgba(53,223,255,0.28);
        border-radius:16px;
        background:linear-gradient(
            135deg,
            rgba(5,24,46,0.96),
            rgba(25,18,54,0.96)
        );
        box-shadow:0 0 20px rgba(53,223,255,0.06);
    ">
        <div style="
            color:#55eaff;
            font-size:11px;
            letter-spacing:1.5px;
            margin-bottom:10px;
        ">
            ◈ AI INSIGHT SUMMARY
        </div>
        <div style="
            color:#d7e8ff;
            font-size:14px;
            line-height:1.7;
        ">
            The model has processed the latest demonstration signals.
            Interpret its output alongside the displayed heart-rate,
            HRV, and simulated stress values. This is an experimental
            software demonstration, not a diagnosis or validated
            health prediction.
        </div>
        <div style="
            color:#a995ff;
            font-size:10px;
            letter-spacing:1px;
            margin-top:12px;
        ">
            MODEL OUTPUT · SYNTHETIC DATA · RESEARCH DEMO
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Prediction Output Visualization
try:
    score_value = float(prediction_score)

    # Support either probability values (0-1) or percentage values (0-100)
    if 0.0 <= score_value <= 1.0:
        score_percent = score_value * 100
    else:
        score_percent = max(0.0, min(100.0, score_value))

    st.markdown("### ◈ Model Output Visualization")

    st.progress(int(score_percent))

    st.caption(
        f"Model output: {score_percent:.1f}% — "
        "experimental synthetic-data result, not a clinical risk score."
    )

except (TypeError, ValueError):
    st.info(
        "The model output is available, but it could not be displayed "
        "as a numeric meter."
    )


# Futuristic System Activity Log
st.markdown(
    """
    <div class="neuro-section-label">
        ◈ SYSTEM ACTIVITY LOG
    </div>
    <div class="neuro-panel">
        <div style="color:#55eaff; margin-bottom:14px;">
            DEMONSTRATION SESSION EVENTS
        </div>
        <div style="border-left:2px solid #35dfff; padding-left:16px;">
            <p style="color:#dff7ff; margin:0 0 5px 0;">
                ● Signal dataset generated
            </p>
            <p style="color:#86a9c4; font-size:12px; margin:0 0 16px 0;">
                Synthetic wearable readings prepared for demonstration.
            </p>
            <p style="color:#dff7ff; margin:0 0 5px 0;">
                ● Trend visualization available
            </p>
            <p style="color:#86a9c4; font-size:12px; margin:0 0 16px 0;">
                Historical and simulated signal trends can be explored.
            </p>
            <p style="color:#dff7ff; margin:0 0 5px 0;">
                ● Model output available
            </p>
            <p style="color:#86a9c4; font-size:12px; margin:0;">
                Experimental output uses synthetic demonstration data.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Signal inputs used by the demonstration model
st.markdown("### Signals Evaluated")

signal_col1, signal_col2, signal_col3 = st.columns(3)

with signal_col1:
    st.metric(
        "Heart Rate",
        f"{float(latest['Heart Rate (bpm)']):.1f} bpm"
    )

with signal_col2:
    st.metric(
        "Heart Rate Variability",
        f"{float(latest['HRV (ms)']):.1f} ms"
    )

with signal_col3:
    st.metric(
        "Simulated Stress Index",
        f"{float(latest['Stress Index (simulated)']):.1f}"
    )

st.caption(
    "These are demonstration inputs. The stress index is simulated, "
    "and the displayed values are not a medical assessment."
)

st.metric(
    "Synthetic pattern score",
    f"{prediction_score * 100:.1f}%"
)


# Simulated Signal Status
st.markdown("### 📊 Simulated Signal Status")

heart_rate_value = float(latest["Heart Rate (bpm)"])
hrv_value = float(latest["HRV (ms)"])
stress_value = float(latest["Stress Index (simulated)"])

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if heart_rate_value >= 100:
        st.warning("Heart rate input: Higher demo range")
    else:
        st.info("Heart rate input: Below demo threshold")

with status_col2:
    if hrv_value < 30:
        st.warning("HRV input: Lower demo range")
    else:
        st.info("HRV input: At or above demo threshold")

with status_col3:
    if stress_value >= 60:
        st.warning("Stress input: Higher simulated range")
    else:
        st.info("Stress input: Below demo threshold")

st.caption(
    "These are illustrative software thresholds for the synthetic demo, "
    "not medical reference ranges or health assessments."
)

# Demo Model Interpretation
st.markdown("### 🧠 Demo Model Interpretation")

if prediction_score >= 0.5:
    st.warning(
        "The demonstration model classified this synthetic input pattern "
        "as matching its event-label category."
    )
else:
    st.success(
        "The demonstration model classified this synthetic input pattern "
        "as not matching its event-label category."
    )

st.caption(
    "This interpretation describes the demo model output only. "
    "It does not indicate whether a person is healthy or unwell."
)

st.caption(
    "This score reflects how the demonstration model classifies "
    "synthetic input patterns. It is not a clinical risk probability "
    "and must not be used for medical decisions."
)


st.caption(
    "The model was trained on generated data with a rule-based teaching label. "
    "This score does not represent the probability of a real health event."
)


# Model Transparency
with st.expander("🔍 Model Training & Limitations"):
    st.markdown("**Training approach**")
    
    st.markdown("**What the model classifies**")
    st.write(
        "The demo model classifies synthetic signal patterns using "
        "labels generated by a predefined rule: heart rate at or above "
        "100 bpm and simulated stress index at or above 60. These are "
        "software demonstration thresholds, not clinical criteria."
    )
    st.write(
        "The displayed score is the model's estimated probability of "
        "the synthetic class-1 label—not the probability of a real "
        "adverse health event."
    )

    st.write(
        "A Random Forest classifier trained on generated synthetic signals "
        "with rule-based demonstration labels."
    )

    st.markdown("**Input features**")
    st.write(
        "- Heart rate (bpm)\n"
        "- Heart rate variability (ms)\n"
        "- Simulated stress index"
    )

    st.markdown("**Important limitations**")
    st.write(
        "- Synthetic training data does not establish real-world accuracy.\n"
        "- Rule-based labels can make demo test results appear unusually high.\n"
        "- The model has not been clinically validated.\n"
        "- Do not use its output for diagnosis, treatment, or medical decisions."
    )


# Digital Twin Signal Summary
st.markdown("### Digital Twin Signal Summary")


st.caption(
    "Latest dashboard snapshot — demonstration data, not a clinical assessment."
)

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown("**Historical reference**")
    st.write("Source: `historical_health_data.csv`")
    st.write("Purpose: Review the supplied historical sample measurements.")

with summary_col2:
    st.markdown("**Current simulated signals**")
    st.write("Source: Synthetic dashboard data")
    st.write("Purpose: Demonstrate how a changing signal stream could be monitored.")

st.warning(
    "Historical samples and simulated signals may not represent the same "
    "person or a clinically comparable time period. Do not interpret "
    "differences as evidence of health deterioration or improvement."
)



# Trend Chart - Interactive Signal Explorer
st.markdown("### ◈ Wearable Signal Trends")

metric = st.selectbox(
    "Choose a signal to explore",
    [
        "Heart Rate (bpm)",
        "HRV (ms)",
        "Sleep (hours)",
        "Stress Index (simulated)"
    ]
)

record_range = st.selectbox(
    "Select history window",
    ["All records", "Last 5 records", "Last 10 records", "Last 20 records"],
    index=0
)

if record_range == "Last 5 records":
    chart_data = health_data.tail(5).copy()
elif record_range == "Last 10 records":
    chart_data = health_data.tail(10).copy()
elif record_range == "Last 20 records":
    chart_data = health_data.tail(20).copy()
else:
    chart_data = health_data.copy()

fig = px.line(
    chart_data,
    x="Time",
    y=metric,
    markers=True,
    title=f"{metric} over time"
)

fig.update_traces(
    line=dict(color="#35DFFF", width=3),
    marker=dict(
        color="#9677FF",
        size=7,
        line=dict(color="#FFFFFF", width=1)
    ),
    hovertemplate=(
        "<b>%{x}</b><br>"
        + metric + ": %{y:.2f}"
        + "<extra></extra>"
    )
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(4,18,36,0.65)",
    font=dict(
        family="Inter, sans-serif",
        color="#C7E8FF",
        size=12
    ),
    title_font=dict(
        family="Orbitron, sans-serif",
        color="#42E8FF",
        size=18
    ),
    margin=dict(l=15, r=15, t=55, b=15),
    hoverlabel=dict(
        bgcolor="#071B31",
        bordercolor="#35DFFF",
        font=dict(color="#FFFFFF")
    ),
    xaxis_title="Time",
    yaxis_title=metric
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(65,150,200,0.14)",
    zeroline=False,
    linecolor="rgba(65,200,255,0.35)"
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(65,150,200,0.14)",
    zeroline=False,
    linecolor="rgba(65,200,255,0.35)"
)

st.plotly_chart(fig, use_container_width=True)


# Interactive Signal Statistics


# Signal Variation Indicator
st.markdown("### ◈ SIGNAL VARIATION")

if not chart_data.empty and metric in chart_data.columns:
    variation_values = pd.to_numeric(
        chart_data[metric],
        errors="coerce"
    ).dropna()

    if len(variation_values) >= 2:
        signal_std = float(variation_values.std())
        signal_mean = float(variation_values.mean())

        st.metric(
            "Standard Deviation",
            f"{signal_std:.2f}"
        )

        if abs(signal_mean) > 1e-9:
            coefficient_variation = (
                signal_std / abs(signal_mean)
            ) * 100

            st.caption(
                f"Coefficient of variation: "
                f"{coefficient_variation:.2f}%"
            )

        st.caption(
            "Higher variation means the selected values are more "
            "spread out around their average. Interpretation depends "
            "on the signal and its units; this is not a health diagnosis."
        )
    else:
        st.info(
            "At least two numeric records are needed to calculate variation."
        )


 

# NeuroRhythm Twin - Demo Runtime Indicator
st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        justify-content:space-between;
        flex-wrap:wrap;
        gap:12px;
        padding:12px 16px;
        margin:18px 0;
        border:1px solid rgba(53,223,255,0.25);
        border-radius:12px;
        background:linear-gradient(
            90deg,
            rgba(7,25,48,0.95),
            rgba(24,18,53,0.95)
        );
    ">
        <div style="
            color:#55eaff;
            font-size:11px;
            letter-spacing:1.5px;
            font-weight:600;
        ">
            ◈ NEURORHYTHM RUNTIME
        </div>
        <div style="
            color:#9fb8d4;
            font-size:11px;
        ">
            <span style="color:#45f0a5;">●</span>
            DEMO INTERFACE ACTIVE
        </div>
        <div style="
            color:#bda8ff;
            font-size:10px;
            letter-spacing:1px;
        ">
            SYNTHETIC DATA MODE
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Futuristic AI System Status Panel
st.markdown(
    """
    <div class="neuro-panel">
        <div class="neuro-panel-header">
            <span>◈ NEURAL ANALYTICS CORE</span>
            <span class="neuro-status">
                <span class="neuro-status-dot"></span>
                OPERATIONAL
            </span>
        </div>
        <div class="neuro-panel-body">
            <div class="neuro-orb">
                <div class="neuro-orb-inner">AI</div>
            </div>
            <div class="neuro-system-text">
                <h3>Digital Twin Intelligence</h3>
                <p>
                    Signal processing interface initialized.
                    Simulated wearable data is available for analysis.
                </p>
                <div class="neuro-progress">
                    <div class="neuro-progress-fill"></div>
                </div>
                <small>DEMO INTERFACE · SYNTHETIC DATA ONLY</small>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Data table
with st.expander("View simulated data table"):
    st.dataframe(health_data, use_container_width=True)

st.caption(
    "Research prototype only. Simulated values cannot assess or diagnose "
    "a person's health. HRV and heart rate vary with many factors."
)
# Personalized Wellbeing Experiment
st.divider()
st.header("🎧 Personalized Wellbeing Experiment")

st.write(
    "Record a focus rating before and after a short session. "
    "This demo collects self-reported ratings; it does not prove that "
    "audio caused a change in focus."
)

with st.form("wellbeing_experiment_form"):
    session_type = st.selectbox(
        "Choose a session",
        [
            "Calm instrumental audio",
            "Nature sounds",
            "Quiet break without audio"
        ]
    )

    focus_before = st.slider(
        "Focus level BEFORE the session (0 = very low, 10 = very high)",
        min_value=0,
        max_value=10,
        value=4,
        key="focus_before"
    )

    focus_after = st.slider(
        "Focus level AFTER the session (0 = very low, 10 = very high)",
        min_value=0,
        max_value=10,
        value=6,
        key="focus_after"
    )

    submitted = st.form_submit_button("Record experiment")

if "experiment_history" not in st.session_state:
    st.session_state.experiment_history = []

if submitted:
    focus_change = focus_after - focus_before

    result = {
        "Session": session_type,
        "Focus Before": focus_before,
        "Focus After": focus_after,
        "Focus Change": focus_change,
        "Activity": activity_context,
        "Profile": profile_name
    }

    st.session_state.experiment_history.append(result)

    st.success("Experiment recorded for this app session!")

    st.subheader("Session Summary")
    st.write(f"**Session:** {session_type}")
    st.write(f"**Focus before:** {focus_before}/10")
    st.write(f"**Focus after:** {focus_after}/10")
    st.metric("Self-reported focus change", f"{focus_change:+d} points")

    st.info(
        "This is one self-reported observation, not a medical assessment. "
        "Repeat sessions and compare results before drawing conclusions."
    )

if st.session_state.experiment_history:
    st.subheader("📊 Experiment History")

    history_df = pd.DataFrame(st.session_state.experiment_history)
    st.dataframe(history_df, use_container_width=True)

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download experiment history (CSV)",
        data=csv_data,
        file_name="neurorhythm_experiment_history.csv",
        mime="text/csv",
        key="download_experiment_history"
    )

# Historical Health Data
st.divider()
st.header("📅 Historical Health Profile")

history_file = Path(__file__).parent / "historical_health_data.csv"

if history_file.exists():
    historical_data = pd.read_csv(history_file)
    historical_data["timestamp"] = pd.to_datetime(
        historical_data["timestamp"]
    )

    st.write(
        "Historical readings from the fictional demo participant. "
        "These values are simulated, not real patient data."
    )

    st.dataframe(
        historical_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Historical Heart Rate")
    st.line_chart(
        historical_data.set_index("timestamp")["heart_rate_bpm"]
    )

    st.subheader("Historical Stress Index")
    st.line_chart(
        historical_data.set_index("timestamp")["stress_index"]
    )
else:
    st.warning(
        "Historical data file not found. "
        "Create historical_health_data.csv in the project folder."
    )


# Historical vs Current Comparison
st.divider()
st.header("📊 Historical vs Current Comparison")

if history_file.exists() and not historical_data.empty:
    latest_history = historical_data.iloc[-1]
    latest_simulated = health_data.iloc[-1]

    comparisons = [
        {
            "Signal": "Heart Rate (bpm)",
            "Historical": float(latest_history["heart_rate_bpm"]),
            "Simulated": float(latest_simulated["Heart Rate (bpm)"]),
            "Unit": "bpm",
        },
        {
            "Signal": "HRV (ms)",
            "Historical": float(latest_history["hrv_ms"]),
            "Simulated": float(latest_simulated["HRV (ms)"]),
            "Unit": "ms",
        },
        {
            "Signal": "Sleep (hours)",
            "Historical": float(latest_history["sleep_hours"]),
            "Simulated": float(latest_simulated["Sleep (hours)"]),
            "Unit": "hours",
        },
        {
            "Signal": "Stress Index",
            "Historical": float(latest_history["stress_index"]),
            "Simulated": float(latest_simulated["Stress Index (simulated)"]),
            "Unit": "points",
        },
    ]

    comparison_df = pd.DataFrame(comparisons)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Signal Comparison")
    chart_df = comparison_df.set_index("Signal")[["Historical", "Simulated"]]
    st.bar_chart(chart_df)

    st.caption(
        "Historical and simulated values are fictional demonstration data. "
        "Differences are descriptive only and do not indicate a diagnosis "
        "or predict a real health event."
    )
else:
    st.warning("Historical data is unavailable or empty.")

    
# Futuristic Dashboard Footer
st.markdown(
    """
    <div style="
        margin-top: 35px;
        padding: 20px 10px;
        border-top: 1px solid rgba(53, 223, 255, 0.3);
        text-align: center;
        background: linear-gradient(
            180deg,
            transparent,
            rgba(8, 28, 52, 0.45)
        );
    ">
        <div style="
            color: #55eaff;
            font-family: 'Orbitron', sans-serif;
            font-size: 11px;
            letter-spacing: 3px;
            margin-bottom: 8px;
        ">
            NEURORHYTHM TWIN
        </div>
        <div style="
            color: #8ba9c3;
            font-size: 12px;
        ">
            DIGITAL HEALTH TWIN · RESEARCH PROTOTYPE
        </div>
        <div style="
            color: #63829e;
            font-size: 10px;
            margin-top: 10px;
        ">
            Synthetic demonstration data only · Not for clinical decisions
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Downloadable AI Evaluation Report
st.markdown("#### ◈ EXPORT MODEL EVALUATION")

evaluation_report = {
    "evaluation_type": "Synthetic demonstration only",
    "test_samples": demo_metrics["test_samples"],
    "accuracy": demo_metrics["accuracy"],
    "precision": positive_metrics["precision"],
    "recall": positive_metrics["recall"],
    "f1_score": positive_metrics["f1-score"],
    "confusion_matrix": demo_metrics["confusion_matrix"],
    "feature_importance": {},
    "limitations": (
        "Metrics measure agreement with synthetic labels generated "
        "by a demonstration rule. They do not establish clinical "
        "accuracy, medical validity, or real-world predictive ability."
    ),
}

evaluation_report_df = pd.DataFrame(
    [
        {"Metric": "Test samples", "Value": evaluation_report["test_samples"]},
        {"Metric": "Accuracy", "Value": evaluation_report["accuracy"]},
        {"Metric": "Precision", "Value": evaluation_report["precision"]},
        {"Metric": "Recall", "Value": evaluation_report["recall"]},
        {"Metric": "F1-score", "Value": evaluation_report["f1_score"]},
        *[
            {"Metric": f"Feature importance: {name}", "Value": value}
            for name, value in evaluation_report.get("feature_importance", {}).items()
        ],
    ]
)

st.download_button(
    label="⬇ Download AI Evaluation Report (CSV)",
    data=evaluation_report_df.to_csv(index=False).encode("utf-8"),
    file_name="neurorhythm_ai_evaluation.csv",
    mime="text/csv",
)

st.caption(
    "The downloaded report contains synthetic-data evaluation results, "
    "not clinical evidence."
)
