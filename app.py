import os
import joblib
import streamlit as st
from features import extract_features

MODEL_PATH = os.path.join("model", "ipsa_random_forest.joblib")

st.set_page_config(page_title="IPSA", page_icon="🔐", layout="centered")
st.title("Intelligent Password Security Analyzer")
st.caption("Prototype: multi-feature analysis + Random Forest classification")

password = st.text_input("Enter a password", type="password")

if password:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["features"]

    feats = extract_features(password)
    X = [[feats[c] for c in feature_names]]
    label = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

    labels = {0: "Weak", 1: "Moderate", 2: "Strong"}
    st.subheader(f"Classification: {labels[label]}")
    st.write("Model probabilities:", {labels[i]: round(float(probs[i]), 3) for i in range(len(probs))})

    st.write("### Detected features")
    st.json(feats)

    recommendations = []
    if feats["length"] < 12:
        recommendations.append("Increase password length.")
    if feats["dictionary_pattern"]:
        recommendations.append("Avoid common dictionary words or predictable substitutions.")
    if feats["sequential_pattern"]:
        recommendations.append("Avoid sequential characters such as abc or 123.")
    if feats["repeated_pattern"]:
        recommendations.append("Avoid repeated-character runs.")
    if feats["keyboard_pattern"]:
        recommendations.append("Avoid common keyboard sequences.")
    if feats["entropy"] < 50:
        recommendations.append("Use a longer and less predictable password.")
    if not recommendations:
        recommendations.append("No major heuristic weakness was detected.")

    st.write("### Recommendations")
    for r in recommendations:
        st.write("•", r)

    st.info("For research reporting, use the metrics produced by train_model.py on an independent test set. Do not copy example/demo metrics into the paper.")
else:
    st.write("Enter a password to analyze it.")
