# IMDB Sentiment Analysis

## Overview
This project performs sentiment analysis on the IMDB Movie dataset using a machine learning model. The analysis classifies movie reviews as either **positive** or **negative** based on textual features. The project includes data preprocessing, model training, and a **Streamlit web application** for visualization and prediction.

## Features
- **Data Cleaning**: Removes HTML tags, converts text to lowercase, removes special characters, and applies lemmatization using spaCy.
- **Machine Learning Model**: Uses a trained **TF-IDF vectorizer** and a **sentiment classification model**.
- **Interactive Visualizations**:
  - Word Cloud for positive and negative reviews.
  - Word frequency distribution.
  - Sentiment distribution plot.
  - Confusion matrix to evaluate model performance.
- **Sentiment Prediction**: Users can input a movie review and receive a sentiment prediction.

## Installation
### Prerequisites
Ensure you have Python 3.7+ installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### Required Libraries
- **pandas**
- **numpy**
- **scikit-learn**
- **spaCy** (with `en_core_web_sm` model)
- **streamlit**
- **seaborn**
- **matplotlib**
- **wordcloud**
- **joblib**

To install spaCy’s language model, run:
```bash
python -m spacy download en_core_web_sm
```

## Usage
### Running the Streamlit App
To launch the sentiment analysis web application, run:
```bash
streamlit run app.py
```

### Predict Sentiment of a Review
1. Enter a movie review in the text box.
2. Click **Analyze**.
3. The model will predict whether the review is **positive** or **negative**.

### Explore Visualizations
- **Word Cloud**: Displays frequently occurring words in positive or negative reviews.
- **Word Frequency**: Shows the most common words in each sentiment class.
- **Sentiment Distribution**: Displays the number of positive and negative reviews.
- **Confusion Matrix**: Evaluates model performance.

## File Structure
```plaintext
├── app.py              # Streamlit app for visualization and prediction
├── cleaning.ipynb      # Notebook for data cleaning and preprocessing
├── data/               # Folder for storing IMDB dataset
├── models/             # Folder for trained model and vectorizer
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
├── requirements.txt    # Required dependencies
└── README.md           # Project documentation
```

## Dataset
The project uses the **IMDB Movie Dataset**, which consists of movie reviews labeled as **positive** or **negative**.

## Acknowledgments
This project was inspired by various NLP techniques and sentiment analysis research. Special thanks to open-source contributors and the Streamlit community.

## Author
**[Esther Njihia](https://github.com/EstherNjihia)**  
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/EstherNjihia/IMDB_sentiment_Analysis)
