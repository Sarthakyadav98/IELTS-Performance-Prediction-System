import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IELTS Band Predictor",
    page_icon="📚",
    layout="centered",
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📚 IELTS Band Score Predictor")
st.markdown(
    "Enter your assessment scores below to get a predicted IELTS band score."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Skill Scores (IELTS 1–9 scale)")
col1, col2 = st.columns(2)

with col1:
    reading   = st.slider("Reading Score",   1.0, 9.0, 5.0, 0.5)
    writing   = st.slider("Writing Score",   1.0, 9.0, 5.0, 0.5)
    listening = st.slider("Listening Score", 1.0, 9.0, 5.0, 0.5)

with col2:
    speaking   = st.slider("Speaking Score",    1.0, 9.0, 5.0, 0.5)
    mock_score = st.slider("Mock Test Score",   1.0, 9.0, 5.0, 0.5)

st.subheader("Learning Behaviour")
col3, col4 = st.columns(2)

with col3:
    attendance    = st.slider("Attendance (%)",     40.0, 100.0, 75.0, 1.0)
    practice_hrs  = st.slider("Practice Hours/week", 0.5, 10.0,  3.0, 0.5)

with col4:
    vocabulary = st.slider("Vocabulary Score (0–100)", 20.0, 100.0, 60.0, 1.0)
    grammar    = st.slider("Grammar Score (0–100)",    20.0, 100.0, 60.0, 1.0)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict My IELTS Band", use_container_width=True, type="primary"):
    # Build base feature dict (matches training column order)
    features = {
        "Reading_Score":         reading,
        "Writing_Score":         writing,
        "Listening_Score":       listening,
        "Speaking_Score":        speaking,
        "Attendance_Percentage": attendance,
        "Practice_Hours":        practice_hrs,
        "Vocabulary_Score":      vocabulary,
        "Grammar_Score":         grammar,
        "Mock_Test_Score":       mock_score,
    }

    # Add composite features (mirrors feature_engineering.py)
    avg_skill      = np.mean([reading, writing, listening, speaking])
    lang_ability   = (vocabulary + grammar) / 2
    engagement_idx = (attendance / 100) * practice_hrs

    features["Avg_Skill_Score"]   = avg_skill
    features["Language_Ability"]  = lang_ability
    features["Engagement_Index"]  = engagement_idx

    input_df = pd.DataFrame([features])
    input_scaled = scaler.transform(input_df)
    predicted_band = float(model.predict(input_scaled)[0])
    predicted_band = round(min(max(predicted_band, 1.0), 9.0), 1)

    # ── Display result ────────────────────────────────────────────────────────
    st.subheader("Prediction")

    if predicted_band >= 7.0:
        level, colour = "Expert / Very Good", "🟢"
    elif predicted_band >= 6.0:
        level, colour = "Competent", "🟡"
    elif predicted_band >= 5.0:
        level, colour = "Modest", "🟠"
    else:
        level, colour = "Limited", "🔴"

    st.metric(label="Predicted IELTS Band", value=f"{predicted_band} / 9.0")
    st.markdown(f"**Level:** {colour} {level}")

    # Breakdown bar
    st.progress(predicted_band / 9.0)

    st.info(
        f"This prediction is based on your skill scores (avg {avg_skill:.1f}), "
        f"vocabulary/grammar ability ({lang_ability:.1f}/100), "
        f"and an engagement index of {engagement_idx:.2f}."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Linear Regression · RMSE 0.158 · R² 0.981 · "
    "Trained on 4,000 synthetic student records"
)
