from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, os

app = Flask(__name__)
CORS(app)

# Model dosyaları aynı klasörde olmalı
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# NOT: Feature sırası önemli!
# [StudyHours, BooksRead, ScreenTime, SleepHours, SelfScore,
#  ActivitySteps, CaloriesBurned, VeryActiveMinutes, SedentaryMinutes]
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = np.array(data["features"], dtype=float).reshape(1, -1)
    features_scaled = scaler.transform(features)
    raw = float(model.predict(features_scaled)[0])

    # ML + senin kurallarının harmanı (örnek, dilersen değiştir)
    study, books, screen, sleep, selfsc, steps, cals, very, sed = features[0]

    rules = 0.0
    rules += study * 0.10           # 6 saat → +0.6 puan
    rules += books * 0.05           # 2 kitap → +0.1 puan
    rules += -screen * 0.05         # 4 saat ekran → -0.2 puan
    if study == 0: rules += -3.0    # hiç çalışmadıysa -3
    if sleep < 5: rules += -2.0     # uyku <5 saat ise -2

    combined = 0.5 * raw + 0.5 * rules
    # %10 üst sınır, %-10 alt sınır (günlük değişim)
    final_pct = max(-10.0, min(10.0, combined * 1.0))

    return jsonify({
        "percentage": round(final_pct, 2),
        "explanation": (
            f"ML tahmini: {raw:.2f} | Kural skoru: {rules:.2f} "
            f"→ Final: %{final_pct:.2f}.\n"
            f"Detay: study={study}, books={books}, screen={screen}, sleep={sleep}, self_score={selfsc}"
        )
    })

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "msg": "SelfStock API is running."})

if __name__ == "__main__":
    # Render/Railway kendi PORT env değişkeni set edebilir
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)