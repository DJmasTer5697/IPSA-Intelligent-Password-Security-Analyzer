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
# Structural Pattern Detector
# --------------------------------------------------
def has_predictable_structural_pattern(password):
    """
    Detect repeated structural blocks.

    Examples:
        Ab1!Ab1!
        Aa12Aa12
        abc123abc123

    A password is NOT flagged merely because its
    character types change between adjacent characters.
    """

    if len(password) < 8:
        return False

    n = len(password)

    # Check whether the complete password is made
    # from the same block repeated at least twice.
    for block_length in range(2, n // 2 + 1):

        if n % block_length != 0:
            continue

        block = password[:block_length]
        repetitions = n // block_length

        if repetitions >= 2 and block * repetitions == password:
            return True

    return False


# --------------------------------------------------
# HARD-CODED RULE ENGINE
# --------------------------------------------------
def evaluate_security_rules(features, password):

    structural_pattern = has_predictable_structural_pattern(password)

    rules = []

    # R1: Minimum password length
    rules.append({
        "Rule ID": "R1",
        "Rule": "Minimum Length",
        "Condition": "Length >= 12",
        "Passed": features["length"] >= 12,
        "Weakness": "Password length is below 12 characters.",
        "Recommendation": "Use a password with at least 12 characters."
    })

    # R2: Character diversity
    rules.append({
        "Rule ID": "R2",
        "Rule": "Character Diversity",
        "Condition": "Diversity >= 0.60",
        "Passed": features["character_diversity"] >= 0.60,
        "Weakness": "Character diversity is relatively low.",
        "Recommendation": "Use a more diverse combination of characters."
    })

    # R3: Repetition ratio
    rules.append({
        "Rule ID": "R3",
        "Rule": "Repetition Ratio",
        "Condition": "Repetition <= 0.20",
        "Passed": features["repetition_ratio"] <= 0.20,
        "Weakness": "Excessive character repetition detected.",
        "Recommendation": "Avoid excessive repetition of characters or patterns."
    })

    # R4: Repeated pattern
    rules.append({
        "Rule ID": "R4",
        "Rule": "Repeated Pattern",
        "Condition": "No repeated pattern",
        "Passed": features["repeated_pattern"] == 0,
        "Weakness": "A repeated character pattern was detected.",
        "Recommendation": "Avoid repeated character patterns."
    })

    # R5: Sequential pattern
    rules.append({
        "Rule ID": "R5",
        "Rule": "Sequential Pattern",
        "Condition": "No sequential pattern",
        "Passed": features["sequential_pattern"] == 0,
        "Weakness": "A predictable sequential pattern was detected.",
        "Recommendation": "Avoid predictable sequences such as 123 or abc."
    })

    # R6: Keyboard pattern
    rules.append({
        "Rule ID": "R6",
        "Rule": "Keyboard Pattern",
        "Condition": "No keyboard pattern",
        "Passed": features["keyboard_pattern"] == 0,
        "Weakness": "A common keyboard pattern was detected.",
        "Recommendation": "Avoid predictable keyboard sequences."
    })

    # R7: Dictionary pattern
    rules.append({
        "Rule ID": "R7",
        "Rule": "Dictionary Pattern",
        "Condition": "No dictionary pattern",
        "Passed": features["dictionary_pattern"] == 0,
        "Weakness": "A dictionary-related pattern was detected.",
        "Recommendation": "Avoid common dictionary words or predictable word patterns."
    })

    # R8: Predictable structural block
    rules.append({
        "Rule ID": "R8",
        "Rule": "Structural Pattern",
        "Condition": "No repeated structural block",
        "Passed": not structural_pattern,
        "Weakness": "A repeated structural block was detected.",
        "Recommendation": "Avoid repeating the same character structure or block."
    })

    return rules


# --------------------------------------------------
# Rule summary
# --------------------------------------------------
def calculate_rule_compliance(rules):

    total_rules = len(rules)
    passed_rules = sum(rule["Passed"] for rule in rules)
    failed_rules = total_rules - passed_rules

    compliance_score = (passed_rules / total_rules) * 100

    return total_rules, passed_rules, failed_rules, compliance_score


# --------------------------------------------------
# Rule vs ML analysis
# --------------------------------------------------
def analyze_rule_ml_relationship(compliance_score, prediction):

    prediction_text = str(prediction).strip().lower()

    # High exposure
    if "high exposure" in prediction_text:

        if compliance_score >= 75:
            return (
                "⚠️ Rule–ML Conflict: The password satisfies most "
                "predefined security rules, but the ML model predicts "
                "High Exposure. This indicates that rule compliance alone "
                "may not fully represent empirical password exposure."
            )

        else:
            return (
                "⚠️ Rule–ML Agreement: The password violates several "
                "predefined security rules and is also classified as "
                "High Exposure by the ML model."
            )

    # Medium exposure
    elif "medium exposure" in prediction_text:

        if compliance_score >= 75:
            return (
                "ℹ️ Partial Rule–ML Conflict: The password satisfies most "
                "predefined rules, but the ML model estimates Medium "
                "Exposure. This suggests that additional character-level "
                "patterns may influence empirical exposure."
            )

        else:
            return (
                "ℹ️ Rule–ML Agreement: The password has multiple rule "
                "violations and is classified as Medium Exposure."
            )

    # Lower exposure
    elif "lower exposure" in prediction_text or prediction_text == "lower":

        if compliance_score >= 75:
            return (
                "✅ Rule–ML Agreement: The password satisfies most "
                "predefined security rules and is classified in the "
                "Lower Exposure tier."
            )

        else:
            return (
                "ℹ️ Rule–ML Difference: Some predefined rules are violated, "
                "while the ML model classifies the password as Lower Exposure. "
                "This demonstrates that deterministic rules and empirical "
                "exposure assessment capture different aspects of password security."
            )

    return (
        "The relationship between deterministic rules and ML prediction "
        "could not be determined."
    )


# --------------------------------------------------
# Recommendation engine
# --------------------------------------------------
def generate_recommendations(features):

    recommendations = []

    if features["length"] < 12:
        recommendations.append(
            "Increase password length to at least 12 characters."
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

    if features["structural_transitions"] == 1:
        recommendations.append(
            "Avoid predictable letter-number-symbol structures."
        )

    if not recommendations:
        recommendations.append(
            "No major weakness was detected by the analyzed security indicators."
        )

    return recommendations


# --------------------------------------------------
# Password improvement guidance
# --------------------------------------------------
def generate_improvement_guidance(features):

    guidance = []

    has_uppercase = features["uppercase"] > 0
    has_lowercase = features["lowercase"] > 0
    has_digits = features["digits"] > 0
    has_special = features["special"] > 0

    # Missing character categories
    if not has_uppercase:
        guidance.append(
            "Add at least one uppercase letter (A–Z)."
        )

    if not has_lowercase:
        guidance.append(
            "Add lowercase letters (a–z)."
        )

    if not has_digits:
        guidance.append(
            "Add numeric characters (0–9)."
        )

    if not has_special:
        guidance.append(
            "Add at least one special character such as @, #, $, or !."
        )

    # General secure construction guidance
    if not has_uppercase or not has_lowercase or not has_digits or not has_special:

        guidance.append(
            "Illustrative format only: combine unrelated words with "
            "uppercase/lowercase letters, digits and special characters. "
            "Do not simply modify an existing password by adding one symbol."
        )

    return guidance


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

        # --------------------------------------------------
        # Extract IPSA engineered features
        # --------------------------------------------------
        features = extract_features(password)

        # --------------------------------------------------
        # Create engineered-feature vector
        # --------------------------------------------------
        feature_vector = np.array(
            [[features[col] for col in feature_columns]],
            dtype=float
        )

        # --------------------------------------------------
        # Scale engineered features
        # --------------------------------------------------
        feature_scaled = scaler.transform(feature_vector)

        # --------------------------------------------------
        # Character-level TF-IDF
        # --------------------------------------------------
        tfidf_vector = tfidf.transform([password])

        # --------------------------------------------------
        # Combined representation
        # --------------------------------------------------
        combined_vector = hstack(
            [feature_scaled, tfidf_vector]
        )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------
        prediction = model.predict(combined_vector)[0]

        # --------------------------------------------------
        # Probability if supported
        # --------------------------------------------------
        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(combined_vector)[0]

        # --------------------------------------------------
        # Rule Engine
        # --------------------------------------------------
        rule_results = evaluate_security_rules(
            features,
            password
        )

        total_rules, passed_rules, failed_rules, compliance_score = (
            calculate_rule_compliance(rule_results)
        )

        # --------------------------------------------------
        # Rule vs ML relationship
        # --------------------------------------------------
        rule_ml_analysis = analyze_rule_ml_relationship(
            compliance_score,
            prediction
        )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------
        st.divider()
        st.header("📊 Analysis Result")

        st.metric(
            label="Predicted Exposure Tier",
            value=str(prediction)
        )

        prediction_lower = str(prediction).lower()

        if "high exposure" in prediction_lower:

            st.error("⚠️ High Exposure")

        elif "medium exposure" in prediction_lower:

            st.warning("⚠️ Medium Exposure")

        elif "lower exposure" in prediction_lower or prediction_lower == "lower":

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

            st.metric(
                "Special Characters",
                features["special"]
            )

            st.metric(
                "Unique Characters",
                features["unique_chars"]
            )

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

            st.metric(
                "Repeated Pattern",
                features["repeated_pattern"]
            )

            st.metric(
                "Sequential Pattern",
                features["sequential_pattern"]
            )

            st.metric(
                "Keyboard Pattern",
                features["keyboard_pattern"]
            )

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
        # HARD-CODED RULE ASSESSMENT
        # --------------------------------------------------
        st.divider()
        st.header("📋 Rule-Based Security Assessment")

        rule_table = []

        for rule in rule_results:

            rule_table.append({
                "Rule ID": rule["Rule ID"],
                "Security Rule": rule["Rule"],
                "Condition": rule["Condition"],
                "Status": "PASS" if rule["Passed"] else "FAIL"
            })

        rule_df = pd.DataFrame(rule_table)

        st.dataframe(
            rule_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # Rule compliance summary
        # --------------------------------------------------
        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Rules Passed",
                f"{passed_rules}/{total_rules}"
            )

        with col2:

            st.metric(
                "Rules Failed",
                failed_rules
            )

        with col3:

            st.metric(
                "Rule Compliance",
                f"{compliance_score:.2f}%"
            )

        # --------------------------------------------------
        # Rule violations
        # --------------------------------------------------
        failed_rule_details = [
            rule for rule in rule_results
            if not rule["Passed"]
        ]

        if failed_rule_details:

            st.subheader("⚠️ Detected Rule Violations")

            for rule in failed_rule_details:

                st.warning(
                    f'{rule["Rule ID"]} — {rule["Weakness"]}'
                )

        else:

            st.success(
                "All predefined security rules were satisfied."
            )

        # --------------------------------------------------
        # Rule vs ML Analysis
        # --------------------------------------------------
        st.divider()
        st.header("🔬 Rule–ML Consistency Analysis")

        st.info(rule_ml_analysis)

        st.write(
            f"**Rule Compliance:** {compliance_score:.2f}%"
        )

        st.write(
            f"**ML Exposure Prediction:** {prediction}"
        )

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
        # Password Improvement Suggestions
        # --------------------------------------------------
        improvement_guidance = generate_improvement_guidance(features)

        if improvement_guidance:

            st.subheader("🛠️ Password Improvement Suggestions")

            for suggestion in improvement_guidance:

                st.info("• " + suggestion)

        else:

            st.success(
                "The password contains uppercase, lowercase, "
                "numeric and special character categories."
            )

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

            st.write(
                f"Deterministic security rules: {total_rules}"
            )