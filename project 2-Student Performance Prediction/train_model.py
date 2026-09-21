import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load cleaned dataset
df = pd.read_csv("data/cleaned_student_performance.csv")

# Features used for prediction
features = [
    "Attendance",
    "Study_Hours",
    "Assignment_Marks",
    "Midterm_Marks"
]

X = df[features]
y = df["Final_Performance"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Training Completed!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save the trained model
joblib.dump(model, "models/student_performance_model.pkl")

print("Model saved successfully!")