from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load trained intent model
model = joblib.load("chatbot_model.pkl")

# Load FAQ dataset
faq_data = pd.read_csv("faq.csv")

# Create FAQ similarity vectorizer
similarity_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

faq_vectors = similarity_vectorizer.fit_transform(
    faq_data["question"]
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "response": "Please enter a question."
        })

    # Convert user's question into TF-IDF
    user_vector = similarity_vectorizer.transform(
        [user_message]
    )

    # Compare with every FAQ question
    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    best_similarity = max(similarities)

    # IMPORTANT:
    # Reject questions that are not sufficiently similar
    # to any question in our FAQ dataset.
    if best_similarity < 0.45:

        return jsonify({
            "response": (
                "I'm sorry, I couldn't understand your question. "
                "Please ask about passwords, accounts, profiles, "
                "support, or working hours."
            ),
            "intent": "unknown",
            "confidence": round(float(best_similarity), 2)
        })

    # Only predict intent when the question is relevant
    intent = model.predict([user_message])[0]

    responses = {

        "password_reset":
            "You can reset your password from the login page by clicking 'Forgot Password'.",

        "account_creation":
            "You can create a new account by selecting the 'Sign Up' option.",

        "contact_support":
            "You can contact our support team for further assistance.",

        "working_hours":
            "Customer support is available during the organization's working hours.",

        "profile_update":
            "You can update your profile information from your account settings.",

        "account_information":
            "You can view your account details from your profile or account settings."
    }

    response = responses.get(
        intent,
        "I'm sorry, I couldn't understand your question."
    )

    return jsonify({
        "response": response,
        "intent": intent,
        "confidence": round(float(best_similarity), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)