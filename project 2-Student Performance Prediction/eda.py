import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/cleaned_student_performance.csv")

print("Cleaned Dataset:")
print(df)

# Basic statistics
print("\nDataset Statistics:")
print(df.describe())

# Chart 1: Study Hours vs Final Performance
plt.figure(figsize=(8, 5))
plt.scatter(df["Study_Hours"], df["Final_Performance"])
plt.xlabel("Study Hours")
plt.ylabel("Final Performance")
plt.title("Study Hours vs Final Performance")
plt.savefig("charts/study_hours_vs_performance.png")
plt.show()

# Chart 2: Attendance vs Final Performance
plt.figure(figsize=(8, 5))
plt.scatter(df["Attendance"], df["Final_Performance"])
plt.xlabel("Attendance")
plt.ylabel("Final Performance")
plt.title("Attendance vs Final Performance")
plt.savefig("charts/attendance_vs_performance.png")
plt.show()

# Chart 3: Final Performance Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Final_Performance"], bins=8)
plt.xlabel("Final Performance")
plt.ylabel("Number of Students")
plt.title("Final Performance Distribution")
plt.savefig("charts/final_performance_distribution.png")
plt.show()

print("\nEDA completed successfully!")
print("Charts saved in the charts folder.")