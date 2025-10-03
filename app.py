from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# 🔹 Eğitimde kullandığın model ve scaler dosyalarını yükle
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# 🔹 Dataset’teki feature kolonları (PerformanceChange hariç)
FEATURE_COLUMNS = [
    "StudyHours",
    "BookPages",
    "Steps",
    "SleepHours",
    "ScreenTime",
    "SelfScore"
]

@app.route("/")
def home():
    return jsonify({"msg": "SelfStock API is running.", "ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # features JSON’dan alınır → DataFrame’e çevrilir
        features = data["features"]
        features_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        # Ölçekleme
        features_scaled = scaler.transform(features_df)

        # Tahmin
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            "percentage": round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
