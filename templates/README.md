# Utility Usage Prediction System

## Project Overview

This project is a Machine Learning based Utility Usage Prediction System.

It predicts utility consumption based on:

- Temperature
- Humidity
- Occupancy

The predicted consumption is classified into:

- Low Usage
- Medium Usage
- High Usage

The system also provides a simple recommendation based on the predicted usage.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Flask
- HTML
- CSS

## Project Structure

CODEVDX INTERNSHIP
│
├── data
│   └── utility_usage.csv
│
├── models
│   └── utility_usage_model.pkl
│
├── templates
│   └── index.html
│
├── app.py
├── main.py
├── predict.py
└── requirements.txt

## How to Run

1. Activate the virtual environment.

2. Install the required libraries:

pip install -r requirements.txt

3. Run the Flask application:

python app.py

4. Open the application in a browser:

http://127.0.0.1:5000

## Features

- Utility usage prediction
- Usage classification
- Personalized recommendation
- Web-based user interface
- Machine Learning model integration