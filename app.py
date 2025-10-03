from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Uygulama başlat
app = Flask(__name__)
CORS(app)

# Model ve scaler yükle
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"msg": "SelfStock API is running.", "ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not isinstance(features, dict):
            return jsonify({"error": "Features must be provided as a dict"}), 400

        # DataFrame'e çevir (kolon isimleri korunur)
        X = pd.DataFrame([features])
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        return jsonify({"percentage": float(round(prediction, 2))})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
