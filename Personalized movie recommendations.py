import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page config
# -----------------------------
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
# Load dataset (manual upload)
# -----------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload the CSV file to proceed.")
        return None

uploaded_file = st.file_uploader(
    "Upload your movie dataset CSV",
    type=["csv"]
)

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
    # TF-IDF Vectorization
    # -----------------------------
    @st.cache_resource
    def build_tfidf(corpus):
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(corpus)
        return vectorizer, tfidf_matrix

    vectorizer, tfidf_matrix = build_tfidf(data['clean_text'])

    # -----------------------------
    # User input
    # -----------------------------
    st.subheader("🔍 Enter Your Movie Preferences")

    user_input = st.text_input(
        "Describe the type of movie you like:",
        placeholder="e.g. action sci-fi space adventure"
    )

    top_n = st.slider("Number of recommendations", 3, 10, 5)

    # -----------------------------
    # Recommendation
    # -----------------------------
    if st.button("Get Recommendations") and user_input.strip():
        user_vector = vectorizer.transform([preprocess_text(user_input)])
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

        top_indices = np.argsort(similarity_scores[0])[::-1][:top_n]

        recommendations = pd.DataFrame({
            "Movie Title": data.loc[top_indices, "title"].values,
            "Similarity Score": similarity_scores[0][top_indices]
        }).sort_values(by="Similarity Score", ascending=False)

        st.subheader("📋 Recommended Movies")
        st.dataframe(recommendations, use_container_width=True)

        st.subheader("📊 Similarity Scores")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            recommendations["Movie Title"],
            recommendations["Similarity Score"]
        )
        ax.invert_yaxis()
        ax.set_xlabel("Cosine Similarity")
        st.pyplot(fig)

    else:
        st.info("Enter your preferences and click **Get Recommendations**")
