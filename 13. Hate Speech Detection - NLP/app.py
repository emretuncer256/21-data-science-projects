import joblib
import streamlit as st

# Load the model
model = joblib.load('hate_speech_detection.pkl')

# Class labels
CLASS_LABELS = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "No Hate and Offensive"
}

# Streamlit app
st.title(":symbols_over_mouth: Hate Speech Detection")
st.write("NLP project to detect hate speech in a text.")
st.image("https://i0.wp.com/cjp.org.in/wp-content/uploads/2018/01/Hate-Speech-FE-Legal-Resource.png?fit=1020%2C534&ssl=1", width='stretch')

# Input text
text = st.text_input("Enter a text")

# Predict button
if st.button("Predict", type="primary", use_container_width=True):
    prediction = model.predict([text])
    st.success(f"Prediction: **{CLASS_LABELS[prediction[0]]}**")