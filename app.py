from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Model ve scaler yükle
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return jsonify({"msg": "SelfStock API is running.", "ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # ✅ 9 feature alıyoruz
        features = np.array([[
            data.get("StudyHours", 0),
            data.get("ExamScore", 0),
            data.get("BooksRead", 0),
            data.get("ScreenTime", 0),
            data.get("SleepHours", 0),
            data.get("SelfScore", 0),
            data.get("ActivitySteps", 0),
            data.get("CaloriesBurned", 0),
            data.get("VeryActiveMinutes", 0)
        ]])

        # Ölçekleme
        features_scaled = scaler.transform(features)

        # Tahmin
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            "percentage": float(prediction),
            "ok": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
