import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="Customer Spend Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #34495e;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #3498db;
        color: white;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .prediction-value {
        font-size: 48px;
        color: #27ae60;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Loading and Model Training
# ==========================================
@st.cache_resource
def load_data_and_train_model():
    # Load dataset
    df = pd.read_csv("E-commerce cleaned.csv")
    
    # We will use base features and let Random Forest handle non-linearities, 
    # dropping the pre-calculated polynomial features to keep input clean.
    features = ['Age', 'Items Purchased', 'Days Since Last Purchase', 
                'Gender', 'City', 'Membership Type', 'Discount Applied', 'Satisfaction Level']
    target = 'Total Spend'
    
    X = df[features]
    y = df[target]
    
    # Define categorical and numerical columns
    categorical_cols = ['Gender', 'City', 'Membership Type', 'Discount Applied', 'Satisfaction Level']
    numerical_cols = ['Age', 'Items Purchased', 'Days Since Last Purchase']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Complete Pipeline with a Random Forest Regressor
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    model.fit(X, y)
    
    return model

# Load model with a loading spinner
with st.spinner("Initializing predictive models..."):
    rf_model = load_data_and_train_model()

# ==========================================
# 3. Application Layout & UI
# ==========================================
st.title("🛍️ E-Commerce Spend Prediction App")
st.markdown("Predict the expected **Total Spend** of a customer based on their demographic and behavioral profiles.")

# Create columns for the layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 📊 Customer Profile")
    st.markdown("Adjust the filters below to simulate a customer.")
    
    # Numerical Inputs (ranges defined based on dataset stats)
    age = st.slider("Age", min_value=18, max_value=70, value=30, step=1)
    items_purchased = st.slider("Items Purchased", min_value=1, max_value=50, value=12, step=1)
    days_since_last = st.slider("Days Since Last Purchase", min_value=0, max_value=100, value=26, step=1)
    
    st.divider()
    
    # Categorical Inputs
    gender = st.selectbox("Gender", ["Female", "Male"])
    membership = st.selectbox("Membership Type", ["Bronze", "Silver", "Gold"], index=1)
    satisfaction = st.selectbox("Satisfaction Level", ["Unsatisfied", "Neutral", "Satisfied"], index=1)
    discount = st.radio("Discount Applied?", [True, False], horizontal=True)
    city = st.selectbox("City", ["New York", "Los Angeles", "Chicago", "San Francisco", "Miami", "Houston"])

with col2:
    st.markdown("### 🔮 Prediction Dashboard")
    st.markdown("The machine learning model uses historical data patterns to estimate revenue.")
    
    # Create an action button
    predict_button = st.button("Calculate Expected Spend")
    
    # Layout for prediction result
    if predict_button:
        # Create a dataframe for the input
        input_data = pd.DataFrame({
            'Age': [age],
            'Items Purchased': [items_purchased],
            'Days Since Last Purchase': [days_since_last],
            'Gender': [gender],
            'Membership Type': [membership],
            'Satisfaction Level': [satisfaction],
            'Discount Applied': [discount],
            'City': [city]
        })
        
        # Predict
        predicted_spend = rf_model.predict(input_data)[0]
        
        # Display nicely
        st.markdown(f"""
            <div class="prediction-box">
                <h4 style="color: #7f8c8d; margin-bottom: 0;">Predicted Total Spendings</h4>
                <div class="prediction-value">${predicted_spend:,.2f}</div>
                <p style="color: #95a5a6;">Based on the provided customer characteristics</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Breakdown section (optional UX enhancement)
        st.write("")
        st.markdown("#### Input Summary")
        summary_cols = st.columns(4)
        summary_cols[0].metric(label="Items", value=items_purchased)
        summary_cols[1].metric(label="Tier", value=membership)
        summary_cols[2].metric(label="Discount", value="Yes" if discount else "No")
        summary_cols[3].metric(label="Sentiment", value=satisfaction)
        
    else:
        st.info("👈 Set the parameters on the left and click **'Calculate Expected Spend'** to see the prediction.")