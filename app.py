from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained ML model
model = joblib.load("models/utility_usage_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    category = None
    recommendation = None

    if request.method == "POST":

        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        occupancy = float(request.form["occupancy"])

        if temperature < 0 or humidity < 0 or occupancy < 0:
            return "Please enter valid positive values."

        # Predict utility usage
        predicted_usage = model.predict(
            [[temperature, humidity, occupancy]]
        )[0]

        # Classify usage
        if predicted_usage < 150:
            category = "Low Usage"
            recommendation = "Usage is currently low."

        elif predicted_usage < 250:
            category = "Medium Usage"
            recommendation = "Consider monitoring your utility consumption."

        else:
            category = "High Usage"
            recommendation = "Consider reducing unnecessary utility usage."

        prediction = round(predicted_usage, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        category=category,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run(debug=True)