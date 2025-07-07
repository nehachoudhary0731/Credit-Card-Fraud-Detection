import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained pipeline
model = joblib.load("credit_card_fraud_model.pkl")

# Page Config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background-color: #0f0f23;
        color: #ffffff;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        margin-bottom: 40px;
    }
    .prediction-box {
        border: 3px solid #00c853;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    .fraud {
        border-color: #d50000;
        background-color: #ffebee;
        color: #b71c1c;
    }
    .legit {
        border-color: #00c853;
        background-color: #e8f5e9;
        color: #1b5e20;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Provide the <b>Hour</b> of transaction and <b>Amount</b>.<br>The model will predict if it\'s <b>Fraudulent</b> or <b>Legitimate</b>.</div>', unsafe_allow_html=True)

# Input fields inside a container
with st.container():
    hour = st.slider("🕒 Transaction Hour (0–23)", min_value=0, max_value=23, value=12)
    amount = st.number_input("💵 Transaction Amount", min_value=0.0, value=100.0, format="%.2f")

# Add a "Clear Input" button
clear_button = st.button("🧹 Clear Input")

if clear_button:
    hour = 12
    amount = 100.0
    st.experimental_rerun()

# Prediction logic
def predict(hour, amount):
    try:
        try:
            df = pd.read_csv("creditcard.csv.zip", encoding='ISO-8859-1', sep=',', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv("creditcard.csv.zip", encoding='ISO-8859-1', sep=';', on_bad_lines='skip')

        df.drop_duplicates(inplace=True)

        median_input = df.drop("Class", axis=1).median().to_dict()
        input_data = pd.DataFrame([{
            **median_input,
            "Hour": hour,
            "LogAmount": np.log1p(amount)
        }])
        input_data.drop(["Time", "Amount"], axis=1, errors="ignore", inplace=True)

        proba = model.predict_proba(input_data)[0][1]
        is_fraud = proba > 0.5

        result = {
            "status": "🚨 Fraudulent Transaction Detected!" if is_fraud else "✅ Legitimate Transaction",
            "emoji": "🚨" if is_fraud else "✅",
            "confidence": f"{proba:.2%}",
            "risk": "🟥 HIGH" if proba > 0.7 else "🟨 MEDIUM" if proba > 0.3 else "🟩 LOW",
            "class": "fraud" if is_fraud else "legit"
        }
        return result

    except Exception as e:
        return {"error": str(e)}

# When user clicks button
if st.button("🔍 Predict Fraud"):
    output = predict(hour, amount)
    if "error" in output:
        st.error(f"Error: {output['error']}")
    else:
        prediction_html = f"""
        <div class="prediction-box {output['class']}">
            {output['status']}<br><br>
            Confidence: {output['confidence']}<br>
            Risk Level: {output['risk']}
        </div>
        """
        st.markdown(prediction_html, unsafe_allow_html=True)