import joblib
import streamlit as st
import pandas as pd

# Load the pipeline
pipeline = joblib.load('credit_card_clustering_pipeline.pkl')

# Streamlit app
st.title(":credit_card: Credit Card Clustering")
st.write("Predict the cluster of a credit card customer based on their features.")
st.image("https://cdn.finshots.app/images/2022/04/Artboard-2b.png", width='stretch')

# Create input form
balance = st.number_input("Balance", min_value=0, max_value=100000, value=2100)
purchases = st.number_input("Purchases", min_value=0, max_value=100000, value=2200)
credit_limit = st.number_input("Credit Limit", min_value=0, max_value=100000, value=3700)

# Predict button
if st.button("Predict Cluster", type="primary", use_container_width=True):
    input_df = pd.DataFrame({
        'BALANCE': [balance],
        'PURCHASES': [purchases],
        'CREDIT_LIMIT': [credit_limit]
    })
    cluster = pipeline.predict(input_df)[0]
    st.success(f"Cluster: **{cluster}**")