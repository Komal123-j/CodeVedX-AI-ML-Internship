import pandas as pd

# Load the datasets
fake_news = pd.read_csv("data/Fake.csv")
real_news = pd.read_csv("data/True.csv")

# Add labels
fake_news["label"] = "Fake"
real_news["label"] = "Real"

# Combine both datasets
news_data = pd.concat([fake_news, real_news], ignore_index=True)

print("===== DATASET INFORMATION =====")
print("Total rows:", len(news_data))
print("Total columns:", len(news_data.columns))

print("\n===== COLUMNS =====")
print(news_data.columns.tolist())

print("\n===== MISSING VALUES =====")
print(news_data.isnull().sum())

print("\n===== FAKE / REAL COUNT =====")
print(news_data["label"].value_counts())

print("\n===== FIRST 5 RECORDS =====")
print(news_data.head())