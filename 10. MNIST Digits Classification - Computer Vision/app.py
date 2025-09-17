import joblib
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = joblib.load('mnist_svc_model.pkl')

# Streamlit app
st.title("🔢 MNIST Digits Classification")
st.write("Predict the digit of a handwritten digit.")
st.image("https://opendatascience.com/wp-content/uploads/2017/05/handwritten.jpg", width='stretch')

# Image input
image = st.file_uploader("Upload an image of a handwritten digit", type=["jpg", "jpeg", "png"])
if image:
    st.image(image, width=200)

# Predict button
if st.button("Predict", type="primary", use_container_width=True):
    if image:
        img = ImageOps.invert(Image.open(image).convert('L').resize((28, 28)))
        x = np.array(img, dtype=np.float32).reshape(1, -1) / 255.0
        prediction = model.predict(x)
        st.success(f"Predicted digit: **{prediction[0]}**")
    else:
        st.error("Please upload an image of a handwritten digit")