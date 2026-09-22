import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# Load datasets
fake_news = pd.read_csv("data/Fake.csv")
real_news = pd.read_csv("data/True.csv")

# Add labels
fake_news["label"] = "Fake"
real_news["label"] = "Real"

# Combine datasets
news_data = pd.concat([fake_news, real_news], ignore_index=True)

# Use title + text as the input
news_data["content"] = news_data["title"] + " " + news_data["text"]

# Load English stopwords
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and very short words
    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    # Join tokens back into text
    return " ".join(tokens)


# Apply preprocessing
news_data["cleaned_content"] = news_data["content"].apply(preprocess_text)

# Display an example
print("===== ORIGINAL TEXT =====")
print(news_data["content"].iloc[0][:500])

print("\n===== CLEANED TEXT =====")
print(news_data["cleaned_content"].iloc[0][:500])

print("\n===== PREPROCESSING COMPLETED =====")
print("Total articles processed:", len(news_data))