# AI Helpdesk Chatbot

## Project Overview

AI Helpdesk Chatbot is a conversational AI application that uses Natural Language Processing (NLP) and Machine Learning to understand user questions and provide appropriate helpdesk responses.

The chatbot identifies the intent of a user's question and responds according to the detected intent.

## Features

- Natural Language Processing
- Intent detection
- TF-IDF text vectorization
- Logistic Regression classification
- FAQ-based training dataset
- Confidence/similarity-based fallback handling
- Flask web application
- Interactive chatbot interface
- Model saving using Joblib
- Input validation
- Unknown-question detection

## Supported Intents

The chatbot currently supports:

- Password Reset
- Account Creation
- Contact Support
- Working Hours
- Profile Update
- Account Information

## Example Questions

### Password Reset

"I forgot my password"

### Account Creation

"How can I create a new account?"

### Contact Support

"How can I contact customer support?"

### Working Hours

"When is customer support available?"

### Profile Update

"How can I update my profile?"

### Account Information

"Where can I see my account details?"

## Unknown Questions

The chatbot also detects questions that are outside its trained FAQ topics.

For example:

"What is the weather in Mumbai?"

The chatbot responds with a fallback message instead of incorrectly assigning the question to another intent.

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- HTML
- CSS
- JavaScript

## Machine Learning Workflow

1. FAQ dataset is created.
2. Text questions are converted into TF-IDF features.
3. Logistic Regression is trained for intent detection.
4. The trained model is saved using Joblib.
5. Flask loads the trained model.
6. User questions are received through the web interface.
7. FAQ similarity is checked.
8. Relevant questions are classified into intents.
9. The chatbot returns the appropriate response.
10. Unrelated questions receive a fallback response.

## Project Structure

```text
Project 4
│
├── .venv
├── templates
│   └── index.html
├── faq.csv
├── train_model.py
├── chatbot_model.pkl
├── app.py
└── README.md