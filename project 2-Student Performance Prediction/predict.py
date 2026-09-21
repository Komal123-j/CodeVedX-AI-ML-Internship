import joblib
import numpy as np

# Load the trained model
model = joblib.load("models/student_performance_model.pkl")

print("===================================")
print("   STUDENT PERFORMANCE PREDICTOR")
print("===================================")

# Take input from the user
attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours per day: "))
assignment_marks = float(input("Enter Assignment Marks: "))
midterm_marks = float(input("Enter Midterm Marks: "))

# Validate inputs
if not 0 <= attendance <= 100:
    print("Invalid attendance. Please enter a value between 0 and 100.")
    exit()

if study_hours < 0:
    print("Invalid study hours. Please enter a positive value.")
    exit()

if not 0 <= assignment_marks <= 100:
    print("Invalid assignment marks. Please enter a value between 0 and 100.")
    exit()

if not 0 <= midterm_marks <= 100:
    print("Invalid midterm marks. Please enter a value between 0 and 100.")
    exit()
# Prepare input for prediction
input_data = np.array([[
    attendance,
    study_hours,
    assignment_marks,
    midterm_marks
]])

# Make prediction
prediction = model.predict(input_data)


# Display result
score = round(prediction[0], 2)

print("\nPredicted Final Performance:", score)

# Performance category
if score >= 75:
    category = "Excellent"
    recommendation = "Great performance! Keep maintaining your study routine."
elif score >= 50:
    category = "Good"
    recommendation = "Good performance. Focus on improving weak areas."
else:
    category = "Needs Improvement"
    recommendation = "Increase study time and focus on assignments and midterm preparation."

print("Performance Category:", category)
print("Recommendation:", recommendation)