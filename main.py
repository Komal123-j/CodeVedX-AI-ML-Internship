import pandas as pd
import joblib

# Load the saved model
model = joblib.load("models/utility_usage_model.pkl")

print("Saved model loaded successfully!")

# Get user input
try:
    temperature = float(input("Enter temperature (°C): "))
    humidity = float(input("Enter humidity (%): "))
    occupancy = int(input("Enter number of people: "))

    # Validate input
    if humidity < 0 or humidity > 100:
        print("Error: Humidity must be between 0 and 100.")
    elif occupancy < 0:
        print("Error: Number of people cannot be negative.")
    else:
        # Create input data
        new_data = pd.DataFrame({
            "temperature": [temperature],
            "humidity": [humidity],
            "occupancy": [occupancy]
        })

        # Make prediction
        prediction = model.predict(new_data)

        print("\nPredicted Utility Usage:", round(prediction[0], 2))

except ValueError:
    print("Error: Please enter valid numbers.")