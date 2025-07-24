from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
#remember to install cors
from flask_cors import CORS
app = Flask(__name__)
#allow CORS
CORS(app)

# Load trained model
model = joblib.load("Datasets/heart_cleveland_upload.csv")

# @app.route("/")
# def home():
#     return render_template("index.html")  # Optional HTML form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
app.run(host="0.0.0.0", port=10000, debug=True)
