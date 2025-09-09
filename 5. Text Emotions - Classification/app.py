import joblib
import pandas as pd
import streamlit as st
import altair as alt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model('model/text_emotions_model.keras')

# Load the tokenizer
tokenizer = joblib.load('model/tokenizer.pkl')

# Load the encoder
encoder = joblib.load('model/encoder.pkl')

# Streamlit app
st.title("Text Emotions Classification")
st.write("Predict the emotions of a text.")
st.image("https://t4.ftcdn.net/jpg/16/58/09/95/360_F_1658099569_2DVa2bX9QN14KmF4c00wmPjIWH6RNDCH.jpg")

# Emoji mapping for classes
EMOJI_BY_CLASS = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😊",
    "love": "❤️",
    "sadness": "😢",
    "surprise": "😲",
}

# Color mapping for classes
EMOTION_COLORS = {
    "anger": "#e74c3c",
    "fear": "#8e44ad",
    "joy": "#f1c40f",
    "love": "#e84393",
    "sadness": "#3498db",
    "surprise": "#2ecc71",
}

# Input text
text = st.text_input("Enter a text")

# Predict emotion probabilities
if text:
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=66)
    prediction = model.predict(padded_sequences, verbose=0)
    probabilities = prediction[0]
    class_names = list(encoder.classes_)

    # Sort emotions by probability descending
    sorted_pairs = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)

    # Top prediction highlight
    top_class, top_prob = sorted_pairs[0]
    top_emoji = EMOJI_BY_CLASS.get(top_class, "🔹")
    st.markdown(f"### {top_emoji} Top emotion: **{top_class}** — {top_prob * 100:.2f}%")

    st.subheader("Emotion probabilities")
    display_names = [f"{EMOJI_BY_CLASS.get(name, '🔹')} {name}" for name, _ in sorted_pairs]
    df = pd.DataFrame({
        "Class": [name for name, _ in sorted_pairs],
        "Emotion": display_names,
        "Probability (%)": [round(p * 100, 2) for _, p in sorted_pairs],
    })
    st.dataframe(df, width='stretch')

    # Optional visualization with fixed colors and sorted order
    df_sorted = df.sort_values(by="Probability (%)", ascending=False)
    color_domain = list(EMOTION_COLORS.keys())
    color_range = list(EMOTION_COLORS.values())
    chart = (
        alt.Chart(df_sorted)
        .mark_bar()
        .encode(
            x=alt.X("Probability (%)", type="quantitative"),
            y=alt.Y("Emotion", type="nominal", sort=df_sorted["Emotion"].tolist()),
            color=alt.Color("Class", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
            tooltip=["Emotion", "Probability (%)"]
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)