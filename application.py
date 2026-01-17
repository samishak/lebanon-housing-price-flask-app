import os
import pickle
import numpy as np
from flask import Flask, render_template, request

# -----------------------------
# App setup
# -----------------------------
application = Flask(__name__)
app = application

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load model & scaler
# -----------------------------
ridge_model = pickle.load(
    open(os.path.join(BASE_DIR, "models", "ridge.pkl"), "rb")
)

standard_scaler = pickle.load(
    open(os.path.join(BASE_DIR, "models", "scaler.pkl"), "rb")
)

# -----------------------------
# Home page
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

# -----------------------------
# Prediction route
# -----------------------------
@app.route("/predictdata", methods=["POST"])
def predict_datapoint():

    # Get form values
    area = float(request.form["area_sqm"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    property_age = int(request.form["property_age_years"])
    distance = float(request.form["distance_to_beirut_km"])
    quality = float(request.form["quality_score"])

    # Order MUST match training
    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        property_age,
        distance,
        quality
    ]])

    # Scale & predict
    input_scaled = standard_scaler.transform(input_data)
    prediction = ridge_model.predict(input_scaled)[0]

    return render_template(
        "home.html",
        result=round(prediction, 2)
    )

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)