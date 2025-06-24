import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------------------
# Load Model Dynamically
# -----------------------------------------
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# -----------------------------------------
# Model Selection
# -----------------------------------------
model_options = {
    "Random Forest": "rf_model.pkl",
    "Logistic Regression": "logreg_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

st.title("🤖 Customer Engagement Predictor (Multi-Model Approach")
st.write("Predict how likely a customer is to engage with a new product using different ML models.")

selected_model_name = st.selectbox("Choose a model to use:", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]
model = load_model(selected_model_path)

# -----------------------------------------
# User Inputs
# -----------------------------------------
st.header("📝 Customer Profile")

age = st.slider("Age", 18, 70, 35)
account_balance = st.number_input("Account Balance ($)", 0, 100000, 25000)
income = st.number_input("Annual Income ($)", 20000, 200000, 60000)
credit_card_usage = st.slider("Credit Card Usage (0.0 to 1.0)", 0.0, 1.0, 0.5)
tenure_years = st.slider("Years with Bank", 1, 15, 5)
loan = st.radio("Has Loan?", ["yes", "no"])
has_mobile_app = st.radio("Uses Mobile App?", ["yes", "no"])

# -----------------------------------------
# Prediction
# -----------------------------------------
if st.button("🔍 Predict Engagement"):
    input_df = pd.DataFrame({
        'age': [age],
        'account_balance': [account_balance],
        'income': [income],
        'credit_card_usage': [credit_card_usage],
        'tenure_years': [tenure_years],
        'loan': [loan],
        'has_mobile_app': [has_mobile_app]
    })

    # Predict probability and class
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("📊 Prediction Result")
    st.metric("Engagement Probability", f"{prob * 100:.2f}%")
    if pred == 1:
        st.success("✅ Likely to engage")
    else:
        st.warning("❌ Unlikely to engage")

    # Pie Chart
    fig, ax = plt.subplots()
    ax.pie([prob, 1 - prob],
           labels=["Engaged", "Not Engaged"],
           autopct='%1.1f%%',
           explode=(0.1, 0),
           colors=['#00C49F', '#FF6361'],
           startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
