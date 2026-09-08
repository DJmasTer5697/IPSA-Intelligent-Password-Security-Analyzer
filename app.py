import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from features import extract_features


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IPSA - Intelligent Password Security Analyzer",
    page_icon="🔐",
    layout="wide"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
MODEL_PATH = "model_v3_fast/ipsa_v3_fast_combined.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


bundle = load_model()

tfidf = bundle["tfidf"]
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]


# --------------------------------------------------
# Recommendation engine
# --------------------------------------------------
def generate_recommendations(features):
    recommendations = []

    if features["length"] < 12:
        recommendations.append(
            "Increase password length. Longer passwords generally provide "
            "a larger search space."
        )

    if features["character_diversity"] < 0.6:
        recommendations.append(
            "Use a more diverse combination of character types."
        )

    if features["repetition_ratio"] > 0.2:
        recommendations.append(
            "Avoid excessive repetition of characters or patterns."
        )

    if features["repeated_pattern"] == 1:
        recommendations.append(
            "Avoid repeated character patterns."
        )

    if features["sequential_pattern"] == 1:
        recommendations.append(
            "Avoid predictable sequential character patterns."
        )

    if features["keyboard_pattern"] == 1:
        recommendations.append(
            "Avoid common keyboard patterns."
        )

    if features["dictionary_pattern"] == 1:
        recommendations.append(
            "Avoid common dictionary words or easily guessable word patterns."
        )

    if not recommendations:
        recommendations.append(
            "No major weakness was detected by the analyzed security indicators."
        )

    return recommendations


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🔐 Intelligent Password Security Analyzer")
st.subheader("IPSA — Password Guessability and Security Assessment")

st.write(
    "Enter a password to analyze its measurable security characteristics, "
    "predict its empirical exposure tier, and receive actionable recommendations."
)

st.warning(
    "For demonstration purposes, use a dummy password. Do not enter a real "
    "personal or account password."
)


# --------------------------------------------------
# Password input
# --------------------------------------------------
password = st.text_input(
    "Enter Password",
    type="password",
    placeholder="Enter a password for analysis..."
)

analyze = st.button("🔍 Analyze Password", type="primary")


# --------------------------------------------------
# Analysis
# --------------------------------------------------
if analyze:

    if not password:
        st.error("Please enter a password first.")

    else:

        # Extract IPSA engineered features
        features = extract_features(password)

        # Create engineered-feature vector
        feature_vector = np.array(
            [[features[col] for col in feature_columns]],
            dtype=float
        )

        # Scale engineered features
        feature_scaled = scaler.transform(feature_vector)

        # Character-level TF-IDF
        tfidf_vector = tfidf.transform([password])

        # Combined representation
        combined_vector = hstack(
            [feature_scaled, tfidf_vector]
        )

        # Prediction
        prediction = model.predict(combined_vector)[0]

        # Probability if supported
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(combined_vector)[0]

        # --------------------------------------------------
        # Result
        # --------------------------------------------------
        st.divider()
        st.header("📊 Analysis Result")

        st.metric(
            label="Predicted Exposure Tier",
            value=str(prediction)
        )

        if str(prediction).lower() == "high exposure":
            st.error("⚠️ High Exposure")

        elif str(prediction).lower() == "medium exposure":
            st.warning("⚠️ Medium Exposure")

        elif str(prediction).lower() == "lower":
            st.success("✅ Lower Exposure")

        # --------------------------------------------------
        # Password characteristics
        # --------------------------------------------------
        st.subheader("🔎 Password Characteristics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Length", features["length"])
            st.metric("Uppercase", features["uppercase"])
            st.metric("Lowercase", features["lowercase"])
            st.metric("Digits", features["digits"])

        with col2:
            st.metric("Special Characters", features["special"])
            st.metric("Unique Characters", features["unique_chars"])
            st.metric(
                "Character Diversity",
                f'{features["character_diversity"]:.2f}'
            )
            st.metric(
                "Entropy",
                f'{features["entropy"]:.2f}'
            )

        with col3:
            st.metric(
                "Repetition Ratio",
                f'{features["repetition_ratio"]:.2f}'
            )
            st.metric("Repeated Pattern", features["repeated_pattern"])
            st.metric("Sequential Pattern", features["sequential_pattern"])
            st.metric("Keyboard Pattern", features["keyboard_pattern"])

        # --------------------------------------------------
        # Additional indicators
        # --------------------------------------------------
        st.subheader("🛡️ Security Indicators")

        indicators = {
            "Dictionary Pattern": features["dictionary_pattern"],
            "Structural Transitions": features["structural_transitions"],
        }

        indicator_df = pd.DataFrame(
            list(indicators.items()),
            columns=["Indicator", "Value"]
        )

        st.table(indicator_df)

        # --------------------------------------------------
        # Model confidence
        # --------------------------------------------------
        if probabilities is not None:

            st.subheader("🤖 Model Prediction Probabilities")

            classes = model.classes_

            probability_df = pd.DataFrame({
                "Exposure Tier": classes,
                "Probability": probabilities
            })

            probability_df["Probability"] = (
                probability_df["Probability"] * 100
            ).round(2)

            st.bar_chart(
                probability_df.set_index("Exposure Tier")
            )

        # --------------------------------------------------
        # Recommendations
        # --------------------------------------------------
        st.subheader("💡 Security Recommendations")

        recommendations = generate_recommendations(features)

        for recommendation in recommendations:
            st.info("• " + recommendation)

        # --------------------------------------------------
        # Technical information
        # --------------------------------------------------
        with st.expander("Technical Analysis"):

            st.write(
                "The prediction uses the trained combined IPSA + "
                "character-level TF-IDF representation."
            )

            st.write(
                f"Model: {type(model).__name__}"
            )

            st.write(
                f"TF-IDF analyzer: {tfidf.analyzer}"
            )

            st.write(
                f"TF-IDF n-gram range: {tfidf.ngram_range}"
            )

            st.write(
                f"Engineered features used: {len(feature_columns)}"
            )
