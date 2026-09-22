# Fake News Detection

## Project Overview

Fake News Detection is a Machine Learning and Natural Language Processing project that predicts whether a given news article or headline is likely to be Fake News or Real News.

The project uses TF-IDF Vectorization and a Machine Learning classification model to analyze the text.

## Features

- News text preprocessing
- TF-IDF text vectorization
- Machine Learning classification
- Fake/Real news prediction
- Model confidence display
- Flask web application
- Input validation
- User-friendly interface

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Flask
- Joblib
- HTML/CSS

## Project Structure

```text
Project 3 - Fake News Detection
│
├── data
│   ├── Fake.csv
│   └── True.csv
│
├── models
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── templates
│   └── index.html
│
├── inspect_data.py
├── preprocess.py
├── train_model.py
├── app.py
├── requirements.txt
└── .gitignore