import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Load Model and Data ---
# Load the trained pipeline
try:
    model = joblib.load('gn_churn_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please run your training script to create 'churn_model.joblib'.")
    st.stop()

# Define the options for the dropdowns
INDUSTRY_OPTIONS = sorted(['Software', 'IT Services', 'Government (US)', 'Government (International)', 'Financial Services', 'Insurance', 'Telecommunication Services', 'Professional Services', 'Energy'])
COUNTRY_OPTIONS = sorted(['United States', 'United Kingdom', 'India', 'Canada', 'Australia', 'Germany'])
SOURCE_OPTIONS = sorted(['Direct Traffic', 'Offline Sources', 'Email Marketing', 'Organic Search', 'Referrals'])
PRIOR_ACCOUNT_OPTIONS = {
    'Yes, had a prior account': 1,
    'No, this is their first account': 0,
    'Unknown': -1
}

# --- App Title and Description ---
st.title("📊 Customer Churn Predictor")
st.markdown("""
This app uses a logistic regression model I trained to predict the likelihood of a GreyNoise customer churning.
Enter the customer's details below and click 'Predict Churn' to see the result.
""")


# --- Prediction Form ---
with st.form("prediction_form"):
    st.header("Enter Customer Details")
    
    # Use columns for a more compact layout
    col1, col2 = st.columns(2)
    
    with col1:
        industry = st.selectbox("Industry", options=INDUSTRY_OPTIONS)
        country = st.selectbox("Country/Region", options=COUNTRY_OPTIONS)
        arr = st.number_input(
            "Annual Recurring Revenue (ARR)",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000,
            help="Enter the customer's total yearly subscription value."
        )

    with col2:
        source = st.selectbox("Latest Source", options=SOURCE_OPTIONS)
        prior_account_label = st.selectbox(
            "Prior Account Status?",
            options=list(PRIOR_ACCOUNT_OPTIONS.keys())
        )
    
    # The submit button for the form
    submitted = st.form_submit_button("Predict Churn")

# --- Prediction Logic ---
if submitted:
    # Map the user-friendly label to the numeric value the model expects
    prior_account_value = PRIOR_ACCOUNT_OPTIONS[prior_account_label]

    # Create a dictionary of the inputs
    data = {
        'Industry': [industry],
        'Country/Region': [country],
        'Latest Source': [source],
        'Had Prior Account': [prior_account_value],
        'ARR': [arr]
    }
    
    # Convert the dictionary to a pandas DataFrame
    input_df = pd.DataFrame(data)

    # Make prediction
    prediction_proba = model.predict_proba(input_df)
    churn_probability = prediction_proba[0][1] # Probability of the '1' class (Churn)

    # Display Prediction
    st.subheader("Prediction Result")
    
    # Use columns for a cleaner layout
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_probability:.2%}",
            delta=f"{'High Risk' if churn_probability > 0.5 else 'Low Risk'}",
            delta_color="inverse"
        )
    
    with res_col2:
        if churn_probability > 0.66:
            st.error("🚨 High Churn Risk: Immediate attention recommended.")
        elif churn_probability > 0.33:
            st.warning("⚠️ Moderate Churn Risk: Consider proactive engagement.")
        else:
            st.success("✅ Low Churn Risk: Customer appears stable.")

    # Show a progress bar for visual effect
    st.progress(churn_probability)
