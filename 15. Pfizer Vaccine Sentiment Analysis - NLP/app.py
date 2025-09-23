import streamlit as st
import nltk
import neattext as ntx
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Ensure VADER lexicon is available
def _ensure_vader_downloaded() -> None:
    nltk.download('vader_lexicon')


@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    _ensure_vader_downloaded()
    return SentimentIntensityAnalyzer()


# Text preprocessing
stemmer = nltk.SnowballStemmer('english')


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ntx.remove_emojis(text)
    text = ntx.remove_emails(text)
    text = ntx.remove_phone_numbers(text)
    text = ntx.remove_urls(text)
    text = ntx.remove_hashtags(text)
    text = ntx.remove_special_characters(text)
    text = ntx.remove_multiple_spaces(text)
    text = ntx.remove_dates(text)
    text = text.strip()
    text = ntx.remove_stopwords(text)
    text = ntx.normalize(text)
    text = " ".join(stemmer.stem(word) for word in text.split())
    return text


def label_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


st.set_page_config(page_title='Pfizer Vaccine Sentiment - VADER', page_icon='💬', layout='centered')

st.title('💬 Sentiment Polarity Detector (VADER)')
st.write('Detect sentiment polarity of a text using VADER.')

default_text = (
    "Pfizer vaccine is doing a great job protecting people, but some side effects worry me."
)

text_input = st.text_area('Enter text to analyze', value=default_text, height=150)

analyze = st.button('Analyze Sentiment', type='primary', use_container_width=True)

if analyze and text_input.strip():
    sia = get_sentiment_analyzer()
    cleaned = clean_text(text_input)
    scores = sia.polarity_scores(cleaned)
    label = label_from_compound(scores.get('compound', 0.0))

    st.subheader('Result')
    st.markdown(f"**Predicted Polarity:** {label}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('🟢 Positive', f"{scores.get('pos', 0.0):.3f}")
    col2.metric('🔴 Negative', f"{scores.get('neg', 0.0):.3f}")
    col3.metric('🟡 Neutral', f"{scores.get('neu', 0.0):.3f}")
    col4.metric('🔵 Compound', f"{scores.get('compound', 0.0):.3f}")

    with st.expander('Show cleaned text'):
        st.code(cleaned)

elif analyze:
    st.warning('Please enter some text to analyze.')
