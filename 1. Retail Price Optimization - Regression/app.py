import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('model/retail_price_model.pkl')

# Load the encoder
encoder = joblib.load('model/encoder.pkl')

# Load the scaler
scaler = joblib.load('model/scaler.pkl')

st.title("Retail Price Optimization - Regression")
st.write("Predict optimal retail prices based on various features.")

# Input features
qty = st.number_input("Quantity", min_value=0, value=1)
lag_price = st.number_input("Lag Price", min_value=0.0, format="%.2f")
unit_price = st.number_input("Unit Price", min_value=0.0, format="%.2f")
customers = st.number_input("Customers", min_value=0, value=1)
comp1_price_diff = st.number_input("Competitor 1 Price Difference", format="%.2f")
comp3_price_diff = st.number_input("Competitor 3 Price Difference", format="%.2f")
comp2_price_diff = st.number_input("Competitor 2 Price Difference", format="%.2f")
s = st.number_input("S", format="%.2f")
product_category_name = st.selectbox(
    "Product Category Name",
    options=encoder.categories_[0].tolist()
)

# Predict and display output
if st.button("Predict Total Price", type="primary"):
    try:
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'qty': [qty],
            'lag_price': [lag_price],
            'unit_price': [unit_price],
            'customers': [customers],
            'comp1_price_diff': [comp1_price_diff],
            'comp2_price_diff': [comp2_price_diff],
            'comp3_price_diff': [comp3_price_diff],
            's': [s],
            'product_category_name': [product_category_name]
        })

        # One-hot encode the categorical variable
        input_encoded = pd.DataFrame(
            encoder.transform(input_data[['product_category_name']]), 
            columns=encoder.get_feature_names_out(['product_category_name'])
        )
        input_encoded.index = input_data.index
        input_data = pd.concat([input_data.drop('product_category_name', axis=1), input_encoded], axis=1)

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Predict using the pre-trained model
        prediction = model.predict(input_scaled)

        if prediction < 0:
            st.warning("Predicted price is negative, please check the input values.")

        st.success(f"Predicted Total Price: ${prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")