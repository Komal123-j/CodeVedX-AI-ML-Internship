import pandas as pd
import joblib

# Load the saved model
model = joblib.load("models/utility_usage_model.pkl")

print("\n===== Utility Usage Prediction Tool =====\n")

try:
    temperature = float(input("Enter temperature (°C): "))
    humidity = float(input("Enter humidity (%): "))
    occupancy = int(input("Enter number of people: "))

    # Validate input
    if temperature < -50 or temperature > 60:
        print("\nError: Temperature must be between -50°C and 60°C.")

    elif humidity < 0 or humidity > 100:
        print("\nError: Humidity must be between 0 and 100.")

    elif occupancy < 0:
        print("\nError: Number of people cannot be negative.")

    else:
        # Create input data
        new_data = pd.DataFrame({
            "temperature": [temperature],
            "humidity": [humidity],
            "occupancy": [occupancy]
        })

        # Make prediction
        prediction = model.predict(new_data)[0]

        # Categorize usage
        if prediction < 150:
            category = "Low Usage"
            recommendation = "Usage is currently low."

        elif prediction < 220:
            category = "Medium Usage"
            recommendation = "Consider monitoring your utility consumption."

        else:
            category = "High Usage"
            recommendation = "Consider reducing unnecessary utility usage."

        # Display result
        print("\n===== Prediction Result =====")
        print(f"Temperature: {temperature} °C")
        print(f"Humidity: {humidity} %")
        print(f"Occupancy: {occupancy} people")
        print(f"Predicted Utility Usage: {prediction:.2f}")
        print(f"Usage Category: {category}")
        print(f"Recommendation: {recommendation}")
        print("=============================")

except ValueError:
    print("\nError: Please enter valid numbers.")