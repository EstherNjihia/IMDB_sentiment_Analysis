import spacy
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
from sklearn.metrics import confusion_matrix
import re

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


@st.cache_data
def load_data():
    df = pd.read_csv("data/IMDB dataset.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

# Load data and model
df = load_data()
model, vectorizer = load_model()

# Preprocess text using spaCy 
@st.cache_data
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize and lemmatize using spaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Preprocess the entire dataset once and cache it
@st.cache_data
def preprocess_dataset(df):
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    return df

df = preprocess_dataset(df)

# Visualization functions
def plot_wordcloud(sentiment, max_words=100):
    text = " ".join(df[df["sentiment"] == sentiment]["cleaned_review"])
    wordcloud = WordCloud(max_words=max_words, background_color="white").generate(text)
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {sentiment.capitalize()} Reviews", fontsize=14)
    st.pyplot(plt)

def plot_word_freq(sentiment, top_n=20):
    from collections import Counter
    
    words = " ".join(df[df["sentiment"] == sentiment]["cleaned_review"]).split()
    word_freq = Counter(words).most_common(top_n)
    words, counts = zip(*word_freq)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), palette="coolwarm")
    plt.title(f"Top {top_n} Words in {sentiment.capitalize()} Reviews", fontsize=14)
    st.pyplot(plt)

def plot_sentiment_distribution():
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["sentiment"], palette="coolwarm")
    plt.title("Sentiment Distribution", fontsize=14)
    st.pyplot(plt)

def plot_confusion_matrix():
    y_test = df["sentiment"].map({"positive": 1, "negative": 0})
    X_test = vectorizer.transform(df["cleaned_review"])
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix", fontsize=14)
    st.pyplot(plt)

# Main app
def main():
    st.title("IMDB Sentiment Analysis Dashboard")
    
    # Welcome message
    st.markdown(
        """
        Welcome to the **IMDB Sentiment Analysis Dashboard**!  
        Use the tabs below to explore visualizations and predict the sentiment of a movie review.
        """
    )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Word Cloud", "Word Frequency", "Sentiment Distribution", "Confusion Matrix"
    ])
    
    # Word Cloud Tab
    with tab1:
        st.subheader("Word Cloud")
        sentiment = st.radio("Select Sentiment:", ["positive", "negative"], key="wordcloud")
        max_words = st.slider("Number of Words:", 50, 500, 100, key="max_words")
        with st.spinner("Generating word cloud..."):
            plot_wordcloud(sentiment, max_words)
    
    # Word Frequency Tab
    with tab2:
        st.subheader("Word Frequency")
        sentiment = st.radio("Select Sentiment:", ["positive", "negative"], key="wordfreq")
        top_n = st.slider("Number of Top Words:", 10, 50, 20, key="top_n")
        with st.spinner("Generating word frequency plot..."):
            plot_word_freq(sentiment, top_n)
    
    # Sentiment Distribution Tab
    with tab3:
        st.subheader("Sentiment Distribution")
        with st.spinner("Generating sentiment distribution plot..."):
            plot_sentiment_distribution()
    
    # Confusion Matrix Tab
    with tab4:
        st.subheader("Confusion Matrix")
        with st.spinner("Generating confusion matrix..."):
            plot_confusion_matrix()
    
    # Add a section for sentiment prediction
    st.header("Predict Sentiment")
    user_input = st.text_area("Enter a movie review:")
    if st.button("Analyze"):
        if user_input:
            with st.spinner("Analyzing sentiment..."):
                processed_input = preprocess_text(user_input)
                vectorized_input = vectorizer.transform([processed_input])
                prediction = model.predict(vectorized_input)[0]
                sentiment_result = "Positive" if prediction == 1 else "Negative"
                st.subheader(f"Predicted Sentiment: {sentiment_result}")
        else:
            st.warning("Please enter a review to analyze.")
    
    # Footer
    st.markdown(
        """
        ---
        **Created by [Esther Njihia](https://github.com/EstherNjihia)**  
        [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/EstherNjihia/IMDB_sentiment_Analysis)
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()