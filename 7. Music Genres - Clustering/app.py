import joblib
import streamlit as st
import pandas as pd

# Load the pipeline
pipeline = joblib.load('music_genres_pipeline.pkl')

# Load the cluster dataframe
cluster_df = pd.read_csv('cluster_df.csv')

# Function to extract categorical choices from the pipeline
def extract_categorical_choices_from_pipeline(pipeline):
    pre = pipeline.named_steps['preprocessor']  # ColumnTransformer
    choices = {}

    for name, trans, cols in pre.transformers_:
        if trans.__class__.__name__ == 'OrdinalEncoder':
            ole = trans
            for col_name, cats in zip(cols, ole.categories_):
                choices[col_name] = list(cats)

    return choices

categorical_choices = extract_categorical_choices_from_pipeline(pipeline)

# Streamlit app
st.title(":notes: Music Genres Clustering")
st.write("Cluster music genres based on acoustic features.")
st.image("https://static.vecteezy.com/system/resources/thumbnails/024/295/098/small_2x/music-notes-background-illustration-ai-generative-free-photo.jpg", width='stretch')

# Create input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Information")
    artist = st.selectbox("Artist", options=categorical_choices['Artist'])
    top_genre = st.selectbox("Top Genre", options=categorical_choices['Top Genre'])
    year = st.number_input("Year", min_value=1950, max_value=2025, value=2015)
    
    st.subheader("Audio Features")
    bpm = st.number_input("Beats Per Minute (BPM)", min_value=15, max_value=350, value=120)
    energy = st.slider("Energy", min_value=0, max_value=100, value=60)
    danceability = st.slider("Danceability", min_value=0, max_value=100, value=50)
    loudness = st.number_input("Loudness (dB)", min_value=-60, max_value=0, value=-10)

with col2:
    st.subheader("Additional Features")
    liveness = st.slider("Liveness", min_value=0, max_value=100, value=20)
    valence = st.slider("Valence", min_value=0, max_value=100, value=25)
    duration = st.number_input("Length (Duration) in seconds", min_value=30, max_value=2000, value=260)
    acousticness = st.slider("Acousticness", min_value=0, max_value=100, value=30)
    speechiness = st.slider("Speechiness", min_value=0, max_value=100, value=5)
    popularity = st.slider("Popularity", min_value=0, max_value=100, value=60)

# Create predict button
if st.button("Predict Cluster", type="primary", use_container_width=True):
    # Prepare input data
    input_data = pd.DataFrame({
        'Artist': [artist],
        'Top Genre': [top_genre],
        'Year': [year],
        'Beats Per Minute (BPM)': [bpm],
        'Energy': [energy],
        'Danceability': [danceability],
        'Loudness (dB)': [loudness],
        'Liveness': [liveness],
        'Valence': [valence],
        'Length (Duration)': [duration],
        'Acousticness': [acousticness],
        'Speechiness': [speechiness],
        'Popularity': [popularity]
    })
    
    # Make prediction
    cluster = pipeline.predict(input_data)[0]
    
    # Display result
    st.success(f":notes: Predicted Cluster: {cluster}")
    
    # Show similar songs from the same cluster
    st.subheader("Similar Songs in Cluster")
    similar_songs = cluster_df[cluster_df['Cluster'] == cluster].sample(10)
    st.dataframe(similar_songs[['Title', 'Artist', 'Top Genre', 'Year']])
