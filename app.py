from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Model ve scaler yükle
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"msg": "✅ SelfStock API is running.", "ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([[
            data["StudyHours"],
            data["BookPages"],
            data["Steps"],
            data["SleepHours"],
            data["ScreenTime"],
            data["SelfScore"]
        ]])

        # Normalize et
        features_scaled = scaler.transform(features)

        # Tahmin yap
        prediction = model.predict(features_scaled)[0]
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
