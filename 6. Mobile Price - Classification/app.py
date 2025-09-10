import joblib
import streamlit as st
import pandas as pd

# Load the pipeline
pipeline = joblib.load('mobile_price_classification.pkl')

# Streamlit app
st.title(":iphone: Mobile Price Classification")
st.write("Predict the price range of a mobile phone based on its features.")
st.image("https://img.freepik.com/premium-vector/cartoon-hand-holding-mobile-smart-phone-with-celebratory-confetti-flying-around-winner-concept_3482-5775.jpg", width='stretch')

# Class legend
CLASS_LABELS = {
    0: "0 (low cost)",
    1: "1 (medium cost)",
    2: "2 (high cost)",
    3: "3 (very high cost)",
}

# Create input form
col1, col2 = st.columns(2)

with col1:
    battery_power = st.number_input(
        "Battery Power (mAh)", min_value=500, max_value=5000, value=1500, step=50
    )
    int_memory = st.number_input(
        "Internal Memory (GB)", min_value=2, max_value=512, value=32, step=2
    )
    mobile_wt = st.number_input(
        "Mobile Weight (g)", min_value=80, max_value=300, value=150, step=1
    )
    px_height = st.number_input(
        "Pixel Height", min_value=0, max_value=2000, value=800, step=10
    )

with col2:
    px_width = st.number_input(
        "Pixel Width", min_value=0, max_value=2500, value=1280, step=10
    )
    ram = st.number_input(
        "RAM (MB)", min_value=256, max_value=8192, value=4096, step=128
    )
    talk_time = st.number_input(
        "Talk Time (hours)", min_value=2, max_value=24, value=15, step=1
    )
    touch_screen = st.selectbox(
        "Touch Screen", options=[0, 1], index=1, help="1 = Yes, 0 = No"
    )

# Create a button to predict the price range
if st.button("Predict Price Range", type="primary", use_container_width=True):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'battery_power': [battery_power],
        'int_memory': [int_memory],
        'mobile_wt': [mobile_wt],
        'px_height': [px_height],
        'px_width': [px_width],
        'ram': [ram],
        'talk_time': [talk_time],
        'touch_screen': [touch_screen]
    })

    # Predict the price range
    prediction = pipeline.predict(input_data)
    predicted_label = int(prediction[0])
    predicted_text = CLASS_LABELS.get(predicted_label, str(predicted_label))

    st.subheader("Prediction")
    st.success(f":moneybag: **{predicted_text}**")

    with st.expander("Class legend", expanded=True):
        st.markdown("""
        - 0 (low cost)
        - 1 (medium cost)
        - 2 (high cost)
        - 3 (very high cost)
        """)
