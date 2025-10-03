from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Flask app
app = Flask(__name__)
CORS(app)

# Model ve scaler yükle
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return "✅ Self-Stock AI API çalışıyor!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Featureları al
        features = [
            float(data.get("StudyHours", 0)),
            float(data.get("BookPages", 0)),
            float(data.get("Steps", 0)),
            float(data.get("SleepHours", 0)),
            float(data.get("ScreenTime", 0)),
            float(data.get("SelfScore", 0)),
        ]

        # NumPy array'e çevir ve ölçekle
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Tahmin yap
        prediction = model.predict(X_scaled)[0]

        return jsonify({"percentage": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
