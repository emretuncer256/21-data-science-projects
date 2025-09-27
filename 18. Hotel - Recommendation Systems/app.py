import streamlit as st
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
import nltk

warnings.filterwarnings('ignore')

# Download required NLTK data
@st.cache_data
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# Download NLTK data on startup
download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Hotel Recommendation System",
    page_icon="🏨",
    layout="wide"
)

# Title and description
st.title("🏨 Hotel Recommendation System")
st.markdown("Find the perfect hotel for your next trip based on your preferences and location!")

# Load data function
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the hotel data"""
    try:
        # Load data
        df = pd.read_csv('Hotel_Reviews.csv')
        
        # Data preprocessing
        df['Hotel_Address'] = df.Hotel_Address.str.replace('United Kingdom', 'UK')
        df['Country'] = df.Hotel_Address.apply(lambda x: x.split()[-1])
        
        # Drop unnecessary columns
        df.drop(columns=[
            'Additional_Number_of_Scoring',
            'Review_Date',
            'Reviewer_Nationality',
            'Negative_Review', 
            'Review_Total_Negative_Word_Counts',
            'Total_Number_of_Reviews', 
            'Positive_Review',
            'Review_Total_Positive_Word_Counts',
            'Total_Number_of_Reviews_Reviewer_Has_Given', 
            'Reviewer_Score',
            'days_since_review', 'lat', 'lng'], 
            inplace=True
        )
        
        # Process tags
        df['Tags'] = df.Tags.apply(lambda x: ''.join(literal_eval(x)))
        df['Tags'] = df.Tags.str.lower()
        df['Country'] = df.Country.str.lower()
        
        # Preprocess all tags once
        df['Processed_Tags'] = df['Tags'].apply(preprocess_text)
        
        # Create TF-IDF vectorizer and fit it
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        tfidf_matrix = vectorizer.fit_transform(df['Processed_Tags'])
        
        # Create country indices mapping
        country_indices = {}
        for country in df['Country'].unique():
            country_indices[country] = df[df['Country'] == country].index.tolist()
        
        return df, vectorizer, tfidf_matrix, country_indices
    except FileNotFoundError:
        st.error("Hotel_Reviews.csv file not found. Please make sure the file is in the same directory as this app.")
        return None, None, None, None

# Text preprocessing function
def preprocess_text(text):
    """Text preprocessing function"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize and filter stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Recommendation function
def recommend_hotels(df, vectorizer, tfidf_matrix, country_indices, location, description, top_n=5):
    """Get hotel recommendations"""
    location = location.lower()
    
    # Check if location exists
    if location not in country_indices:
        return pd.DataFrame(columns=['Hotel_Name', 'Average_Score', 'Hotel_Address'])
    
    # Preprocess description
    processed_description = preprocess_text(description)
    
    # Transform description
    description_vector = vectorizer.transform([processed_description])
    
    # Get country indices
    country_idx = country_indices[location]
    
    # Calculate similarities
    similarities = cosine_similarity(description_vector, tfidf_matrix[country_idx]).flatten()
    
    # Create results dataframe
    country_df = df.iloc[country_idx].copy()
    country_df['similarity'] = similarities
    
    # Sort and deduplicate
    country_df = country_df.sort_values('similarity', ascending=False)
    country_df = country_df.drop_duplicates(subset=['Hotel_Name'], keep='first')
    country_df = country_df.sort_values(['similarity', 'Average_Score'], ascending=[False, False])
    
    # Return top recommendations
    rec = country_df[['Hotel_Name', 'Average_Score', 'Hotel_Address']].head(top_n)
    rec.sort_values('Average_Score', ascending=False, inplace=True)
    rec.reset_index(drop=True, inplace=True)
    
    return rec

# Main app
def main():
    # Load data
    with st.spinner("Loading and preprocessing hotel data..."):
        df, vectorizer, tfidf_matrix, country_indices = load_and_preprocess_data()
    
    if df is None:
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Search Preferences")
        
        # Country selection
        countries = sorted(df['Country'].unique())
        selected_country = st.selectbox(
            "Select Country:",
            countries,
            index=countries.index('uk') if 'uk' in countries else 0
        )
        
        # Description input
        description = st.text_area(
            "Describe your ideal hotel:",
            placeholder="e.g., Family vacation with kids, need pool and activities",
            height=100
        )
        
        # Number of recommendations
        top_n = st.slider("Number of recommendations:", 1, 10, 5)
        
        # Search button
        search_button = st.button("🔍 Find Hotels", type="primary")
    
    with col2:
        st.header("Recommendations")
        
        if search_button:
            if not description.strip():
                st.warning("Please enter a description of your ideal hotel.")
            else:
                # Get recommendations using preprocessed data
                recommendations = recommend_hotels(
                    df, vectorizer, tfidf_matrix, country_indices,
                    selected_country, description, top_n
                )
                
                if recommendations.empty:
                    st.error(f"No hotels found for '{selected_country}'. Please try a different country.")
                else:
                    # Display recommendations
                    for i, (_, hotel) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            st.markdown(f"### {i}. {hotel['Hotel_Name']}")
                            st.markdown(f"**Rating:** :yellow-badge[{hotel['Average_Score']:.1f} ⭐]")
                            st.markdown(f"**Address:** {hotel['Hotel_Address']}")
                            st.markdown("---")
        else:
            st.info("👈 Enter your preferences and click 'Find Hotels' to get recommendations!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This hotel recommendation system uses:
        - **TF-IDF Vectorization** for text analysis
        - **Cosine Similarity** for matching preferences
        - **NLP Processing** for better recommendations
        
        Enter a description of your ideal hotel and select a country to get personalized recommendations!
        """)
        
        st.header("Example Descriptions")
        st.markdown("""
        - "Family vacation with kids, need pool and activities"
        - "Business trip, need good wifi and central location"
        - "Romantic getaway, luxury hotel with spa"
        - "Budget travel, clean and comfortable"
        - "Pet-friendly hotel near attractions"
        """)

if __name__ == "__main__":
    main()
