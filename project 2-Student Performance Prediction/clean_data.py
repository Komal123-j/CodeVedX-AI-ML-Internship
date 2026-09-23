import pandas as pd

# Load the student dataset
df = pd.read_csv("data/student_performance.csv")

print("Original Dataset:")
print(df)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing numerical values with the column mean
numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# Display cleaned dataset
print("\nCleaned Dataset:")
print(df)

# Save the cleaned dataset
df.to_csv("data/cleaned_student_performance.csv", index=False)

print("\nCleaned dataset saved successfully!")