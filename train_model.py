import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib


# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


# =========================
# 1. LOAD DATA
# =========================

fake_news = pd.read_csv("data/Fake.csv")
real_news = pd.read_csv("data/True.csv")

fake_news["label"] = 0
real_news["label"] = 1

news_data = pd.concat([fake_news, real_news], ignore_index=True)


# =========================
# 2. CREATE CONTENT COLUMN
# =========================

news_data["content"] = (
    news_data["title"].fillna("") + " " +
    news_data["text"].fillna("")
)


# =========================
# 3. TEXT PREPROCESSING
# =========================

stop_words = set(stopwords.words("english"))


def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)


print("Preprocessing news articles...")

news_data["cleaned_content"] = news_data["content"].apply(preprocess_text)


# =========================
# 4. FEATURES AND LABEL
# =========================

X = news_data["cleaned_content"]
y = news_data["label"]


# =========================
# 5. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 6. TF-IDF VECTORIZATION
# =========================

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# =========================
# 7. TRAIN MODEL
# =========================

print("Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)


# =========================
# 8. MAKE PREDICTIONS
# =========================

y_pred = model.predict(X_test_tfidf)


# =========================
# 9. MODEL EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL RESULTS =====")
print("Accuracy:", accuracy)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake", "Real"]
))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))


# =========================
# 10. SAVE MODEL + VECTORIZER
# =========================

joblib.dump(model, "models/fake_news_model.pkl")

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n===== MODEL SAVED =====")
print("Model: models/fake_news_model.pkl")
print("Vectorizer: models/tfidf_vectorizer.pkl")