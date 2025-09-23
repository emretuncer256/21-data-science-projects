import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import neattext as ntx


# -----------------------------
# Helpers
# -----------------------------

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ntx.remove_stopwords(text)
    text = ntx.remove_multiple_spaces(text)
    text = ntx.normalize(text)
    return text


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df = df.dropna(subset=["Title", "Article"]).copy()
    df["article_clean"] = df["Article"].apply(clean_text)
    return df


@st.cache_resource(show_spinner=False)
def build_vectorizer(corpus: pd.Series):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def find_similar_by_text(
    input_text: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    article_matrix,
    top_n: int = 5,
):
    query_clean = clean_text(input_text)
    if not query_clean.strip():
        return []
    query_vec = vectorizer.transform([query_clean])
    sims = cosine_similarity(query_vec, article_matrix).flatten()
    top_idx = np.argsort(-sims)[:top_n]
    results = []
    for i in top_idx:
        results.append({
            "title": df.iloc[i]["Title"],
            "article": df.iloc[i]["Article"],
            "score": float(sims[i]),
        })
    return results


def find_similar_by_title(
    title: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    article_matrix,
    top_n: int = 5,
):
    matches = df.index[df["Title"] == title].tolist()
    if matches:
        idx = matches[0]
        vec = article_matrix[idx]
        sims = cosine_similarity(vec, article_matrix).flatten()
        sims[idx] = -np.inf
        top_idx = np.argsort(-sims)[:top_n]
        results = []
        for i in top_idx:
            results.append({
                "title": df.iloc[i]["Title"],
                "article": df.iloc[i]["Article"],
                "score": float(sims[i]),
            })
        return results
    return find_similar_by_text(title, df, vectorizer, article_matrix, top_n)


# -----------------------------
# UI
# -----------------------------

def main():
    st.set_page_config(page_title="Article Recommender", page_icon="📰", layout="centered")
    st.title("📰 Article Recommendation System")
    st.caption("Content-based recommendations using TF-IDF and cosine similarity")

    df = load_data("https://raw.githubusercontent.com/amankharwal/Website-data/master/articles.csv")
    vectorizer, article_matrix = build_vectorizer(df["article_clean"]) 

    with st.sidebar:
        st.header("Settings")
        top_n = st.number_input("Top N results", min_value=1, max_value=20, value=5, step=1)

    mode = st.radio("Input type", ("Title", "Article content"))

    results = []

    if mode == "Title":
        title = st.selectbox("Select title", options=sorted(df["Title"].unique().tolist()))
        if title:
            st.subheader(title)
            st.write(df.loc[df["Title"] == title, "Article"].iloc[0])
        if st.button("Find similar articles", type="primary"):
            results = find_similar_by_title(title, df, vectorizer, article_matrix, top_n)
    else:
        article_text = st.text_area("Paste article content", height=200)
        if st.button("Find similar articles", type="primary"):
            if not article_text.strip():
                st.warning("Please paste some article content.")
            else:
                results = find_similar_by_text(article_text, df, vectorizer, article_matrix, top_n)

    if results:
        st.subheader("Similar Articles")
        for i, item in enumerate(results, start=1):
            with st.expander(f"{i}. {item['title']}  :yellow-badge[:material/star: {item['score']:.3f}]", expanded=False):
                st.write(item["article"])


if __name__ == "__main__":
    main()
