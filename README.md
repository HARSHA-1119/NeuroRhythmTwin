# 🧠 NeuroRhythm Twin
### A Personalized Digital Health Twin with Adaptive Wellbeing Experiments

## Overview

NeuroRhythm Twin is a software prototype exploring how historical health information, simulated wearable signals, and user-reported focus feedback can be combined in a personal digital twin.

The goal is to help users explore patterns in their wellbeing and evaluate whether different audio or relaxation sessions are associated with changes in their self-reported focus.

## Problem

Health and wellbeing signals vary over time, while generic wellness recommendations may not reflect an individual's patterns or preferences. Users need understandable ways to explore their own trends and evaluate which low-risk wellbeing activities work best for them.

## Proposed Solution

NeuroRhythm Twin brings together:

- Historical health-data visualization
- Simulated wearable readings
- Heart rate, heart-rate variability (HRV), sleep, and a synthetic stress index
- A machine-learning demonstration using synthetic data
- Before-and-after focus ratings for audio and quiet-break experiments
- Historical-versus-simulated signal comparisons

## Current Prototype Features

- Interactive Streamlit dashboard
- Synthetic health-data generation
- Historical CSV data import
- Interactive signal-trend charts
- Demonstration machine-learning score
- User-reported focus experiment form
- Session-level experiment history and CSV export

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## Run Locally

1. Install Python.
2. Clone or download this repository.
3. Open a terminal in the project directory.
4. Create and activate a virtual environment.
5. Install dependencies:

   `python -m pip install -r requirements.txt`

6. Start the application:

   `python -m streamlit run app.py`

## Data and Safety

The current prototype uses fictional historical records and simulated wearable data. Its machine-learning demonstration is trained on synthetic, rule-generated labels and has not been clinically validated.

The displayed model score is not a clinical risk probability, diagnosis, or medical recommendation. Self-reported focus changes do not prove that audio caused an improvement.

Do not upload real patient information to the public repository. Any future use of real health data would require appropriate consent, privacy safeguards, data-quality checks, clinical oversight, and rigorous model evaluation.

## Roadmap

- Persistent storage for experiment history
- Clearly defined prediction target and evaluation on held-out data
- Model explanations and uncertainty reporting
- Optional integration with supported wearable data sources
- Privacy controls and clinician-oriented reporting

## Project Status

Early-stage proof of concept for a digital health twin hackathon. Features and predictive claims remain experimental.