# Student Performance Prediction

## Project Overview

Student Performance Prediction is a Machine Learning project that predicts a student's final performance based on academic and attendance-related factors.

## Features

- Data cleaning using Pandas
- Exploratory Data Analysis (EDA)
- Data visualization
- Linear Regression model
- Student performance prediction
- Performance classification
- Personalized recommendation
- Input validation

## Input Features

The model uses:

- Attendance
- Study Hours
- Assignment Marks
- Midterm Marks

## Output

The system predicts:

- Final Performance score
- Performance Category
- Recommendation for the student

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Structure

```text
project 2-Student Performance Prediction/
│
├── data/
│   ├── student_performance.csv
│   └── cleaned_student_performance.csv
│
├── models/
│   └── student_performance_model.pkl
│
├── charts/
│   ├── attendance_vs_performance.png
│   ├── study_hours_vs_performance.png
│   └── final_performance_distribution.png
│
├── clean_data.py
├── eda.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md