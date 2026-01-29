import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import requests

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CONFIG
# -----------------------------
TMDB_API_KEY = "YOUR_TMDB_API_KEY_HERE"

st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Personalized Movie Recommendation System")
st.write("Content-based recommendation using TF-IDF and Cosine Similarity")

# -----------------------------
# Download NLTK resources
# -----------------------------
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk()

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload the CSV file to proceed.")
        return None

uploaded_file = st.file_uploader("Upload your movie dataset CSV", type=["csv"])
data = load_data(uploaded_file)

if data is not None:

    # -----------------------------
    # Text preprocessing
    # -----------------------------
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    data['clean_text'] = data['overview'].fillna("").apply(preprocess_text)

    # -----------------------------
    # TF-IDF
    # -----------------------------
    @st.cache_resource
    def build_tfidf(corpus):
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(corpus)
        return vectorizer, tfidf_matrix

    vectorizer, tfidf_matrix = build_tfidf(data['clean_text'])

    # -----------------------------
    # USER INPUT
    # -----------------------------
    st.subheader("🔍 Enter Your Movie Preferences")

    user_input = st.text_input("Describe the type of movie you like:",
                               placeholder="e.g. action sci-fi space adventure")

    top_n = st.slider("Number of recommendations", 3, 10, 5)

    # -----------------------------
    # POSTER FUNCTION
    # -----------------------------
    def get_poster(movie_id):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
        return None

    # -----------------------------
    # RECOMMENDATION
    # -----------------------------
    if st.button("Get Recommendations") and user_input.strip():
        user_vector = vectorizer.transform([preprocess_text(user_input)])
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

        top_indices = np.argsort(similarity_scores[0])[::-1][:top_n]

        recommendations = data.loc[top_indices].copy()
        recommendations["Similarity Score"] = similarity_scores[0][top_indices]
        recommendations = recommendations.sort_values(by="Similarity Score", ascending=False)

        st.subheader("📋 Recommended Movies")

        # 🔹 Selectbox
        selected_movie = st.selectbox("🎯 Select a movie to view details:",
                                      recommendations["title"])

        selected_row = recommendations[recommendations["title"] == selected_movie].iloc[0]

        # 🔹 Poster
        poster_url = get_poster(selected_row["id"])
        if poster_url:
            st.image(poster_url, width=220)

        st.markdown(f"### 🎬 {selected_row['title']}")
        st.write(f"**Genre:** {selected_row['genre']}")
        st.write(f"**Overview:** {selected_row['overview']}")
        st.write(f"**Similarity Score:** {selected_row['Similarity Score']:.3f}")

        # 🔹 WHY THIS MOVIE
        feature_names = vectorizer.get_feature_names_out()
        user_vec = user_vector.toarray()[0]
        top_terms = [feature_names[i] for i in user_vec.argsort()[-6:][::-1]]

        st.info(f"🧠 Why this movie? Top matching keywords: **{', '.join(top_terms)}**")

        # 🔹 Show all recommendations
        st.subheader("📌 All Recommendations")

        for _, row in recommendations.iterrows():
            st.markdown(f"### 🎞 {row['title']}")
            st.write(f"**Genre:** {row['genre']}")
            st.write(row["overview"])
            st.write(f"Similarity Score: `{row['Similarity Score']:.3f}`")
            st.divider()

    else:
        st.info("Enter your preferences and click **Get Recommendations**")
