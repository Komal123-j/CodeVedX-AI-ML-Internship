from flask import Flask, render_template, request

import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download required NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


# Create Flask application
app = Flask(__name__)


# Load trained model and TF-IDF vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Load English stopwords
stop_words = set(stopwords.words("english"))


# Text preprocessing function
def preprocess_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and short words
    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)


# Home page
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    message = None

    if request.method == "POST":

        news_text = request.form.get("news_text", "").strip()

        # Input validation
        if not news_text:
            message = "Please enter some news text."

        elif len(news_text) < 20:
            message = "Please enter a longer news article or headline."

        else:

            # Preprocess input
            cleaned_text = preprocess_text(news_text)

            # Convert text into TF-IDF features
            text_vector = vectorizer.transform([cleaned_text])

            # Make prediction
            result = model.predict(text_vector)[0]

            # Get probability/confidence
            probabilities = model.predict_proba(text_vector)[0]
            confidence = max(probabilities) * 100

            if result == 1:
                prediction = "REAL"
            else:
                prediction = "FAKE"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)