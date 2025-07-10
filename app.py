import streamlit as st
import numpy as np
import joblib
import os

#  PAGE CONFIG 
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

#  LOAD MODEL (which I already dump)
MODEL_PATH = 'credit_card_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("❌ Model file not found. Please ensure 'credit_card_model.pkl' exists.")
    st.stop()

#  SESSION STATE SETUP 
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

#  CLEAR BEFORE WIDGET RENDERS 
if st.session_state.clear_input:
    st.session_state.user_input = ""
    st.session_state.clear_input = False

#  CUSTOM STYLING 
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');

        .stTextInput>div>div>input {
            border: 2px solid #FFA3A3;
            border-radius: 8px;
        }

        .title-style {
            font-family: 'Poppins', sans-serif;
            font-size: 42px;
            font-weight: 600;
            background: linear-gradient(to right, #3b82f6, #8b5cf6);  /* Blue to Purple */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle-style {
            text-align: center;
            color: #cbd5e1;
            font-size: 16px;
            margin-bottom: 30px;
        }

        .result-box {
            border: 2px solid #10b981;
            padding: 1rem;
            border-radius: 12px;
            background-color: rgba(16, 185, 129, 0.05);
            text-align: center;
            margin-top: 20px;
        }

        .fraud-box {
            border: 2px solid #ef4444;
            padding: 1rem;
            border-radius: 12px;
            background-color: rgba(239, 68, 68, 0.05);
            text-align: center;
            margin-top: 20px;
        }

        .predict-label {
            font-size: 18px;
            color: #94a3b8;
            font-weight: bold;
        }

        .predict-text {
            font-size: 22px;
            font-weight: bold;
        }

        .green-text {
            color: #10b981;
        }

        .red-text {
            color: #ef4444;
        }

        .button-row {
            display: flex;
            justify-content: center;
            gap: 1rem;
        }

    </style>
""", unsafe_allow_html=True)

#  TITLE 
st.markdown('<div class="title-style">Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-style">Detect fraudulent transactions in real time using AI 🔍</div>', unsafe_allow_html=True)

#  INSTRUCTIONS 
with st.expander("📘 Instructions", expanded=False):
    st.markdown("• Enter exactly **29 comma-separated numerical values** (the transaction features).")

# INPUT SECTION 
st.markdown("### 🔢 Enter Transaction Data")
user_input = st.text_input(
    label="",
    value=st.session_state.user_input,
    key="user_input",
    placeholder="-1.23, 2.45, ..., 85.90"
)

#  BUTTONS 
col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔍 Predict Result")
with col2:
    if st.button("🧹 Clear Input"):
        st.session_state.clear_input = True
        st.rerun()

#  HANDLE PREDICTION 
if predict_btn:
    try:
        input_list = [float(i.strip()) for i in st.session_state.user_input.split(',')]
        if len(input_list) != 29:
            st.error("❗ Please enter exactly 29 values.")
        else:
            input_array = np.array([input_list])
            prediction = model.predict(input_array)

            if prediction[0] == 1:
                st.markdown("""
                    <div class="fraud-box">
                        <div class="predict-label">Prediction:</div>
                        <div class="predict-text red-text">🚨 Fraudulent Transaction</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="result-box">
                        <div class="predict-label">Prediction:</div>
                        <div class="predict-text green-text">✅ Legitimate Transaction</div>
                    </div>
                """, unsafe_allow_html=True)

            # prediction probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_array)[0]
                st.progress(proba[1], text=f"Fraud Probability: {proba[1]:.2%}")
                st.caption(f"Legitimate: {proba[0]:.2%} | Fraudulent: {proba[1]:.2%}")

    except ValueError:
        st.error("❌ Invalid input. Ensure all values are numeric and comma-separated.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
