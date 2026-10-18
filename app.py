import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prediction_model import predict_demo_event
from pathlib import Path

st.set_page_config(
    page_title="NeuroRhythm Twin",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroRhythm Twin")
st.caption(
    "Digital health twin prototype | "
    "Historical sample data and simulated current signals"
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


st.metric(
    "Synthetic pattern score",
    f"{prediction_score * 100:.1f}%"
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


# Trend chart
st.markdown("### Wearable Signal Trends")

metric = st.selectbox(
    "Choose a signal to explore",
    [
        "Heart Rate (bpm)",
        "HRV (ms)",
        "Sleep (hours)",
        "Stress Index (simulated)"
    ]
)

fig = px.line(
    health_data,
    x="Time",
    y=metric,
    markers=True,
    title=f"{metric} over time"
)

fig.update_layout(
    xaxis_title="Time",
    yaxis_title=metric
)

st.plotly_chart(fig, use_container_width=True)

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