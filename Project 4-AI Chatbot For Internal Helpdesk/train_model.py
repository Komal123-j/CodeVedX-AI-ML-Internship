import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load FAQ dataset
data = pd.read_csv("faq.csv")

# Input questions and intent labels
X = data["question"]
y = data["intent"]

# Create NLP pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True)),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "chatbot_model.pkl")

print("Chatbot model trained successfully!")
print("Model saved as chatbot_model.pkl")