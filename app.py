"""
app.py — Flask tabanlı Sahte Haber Tespit API'si (Bölüm 4.4)

Çalıştırmak için:
    1. python train_model.py   (klasik modelleri eğit)
    2. python app.py           (sunucuyu başlat)
BiLSTM için faz2_bilstm.ipynb çalıştırılınca otomatik yüklenir.
"""

import os
import time
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib

from cleaner import clean_tweet, extract_features
from bilstm_model import load_bilstm, bilstm_predict

app = Flask(__name__)
CORS(app)

# ─── Klasik model yükleme ──────────────────────────────────────────────────
_MODELS = {}

_MODEL_NAMES = {
    "nb":  "Naive Bayes",
    "rf":  "Random Forest",
    "lr":  "Lojistik Regresyon",
    "svm": "SVM",
}

def _load_classic_models():
    for key, label in _MODEL_NAMES.items():
        path = os.path.join("models", f"{key}_pipeline.joblib")
        if os.path.exists(path):
            _MODELS[key] = joblib.load(path)
            print(f"[app] {label} modeli yüklendi.")
    if not _MODELS:
        print("[app] UYARI: Klasik model bulunamadı. 'python train_model.py' çalıştırın.")


# ─── BiLSTM model yükleme ─────────────────────────────────────────────────
_bilstm_model  = None
_bilstm_vocab  = None
_bilstm_maxlen = None
_bilstm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_bilstm():
    global _bilstm_model, _bilstm_vocab, _bilstm_maxlen
    path = os.path.join("models", "bilstm_model.pt")
    m, v, ml = load_bilstm(path, _bilstm_device)
    if m is not None:
        _bilstm_model, _bilstm_vocab, _bilstm_maxlen = m, v, ml
        print("[app] BiLSTM modeli yüklendi.")
    else:
        print("[app] BiLSTM modeli bulunamadı (faz2_bilstm.ipynb çalıştırılınca yüklenir).")


# ─── BerTURK model yükleme ────────────────────────────────────────────────
_berturk_model     = None
_berturk_tokenizer = None
_berturk_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_berturk():
    global _berturk_model, _berturk_tokenizer
    path = os.path.join("models", "berturk")
    if not os.path.exists(path):
        print("[app] BerTURK modeli bulunamadı (faz2_berturk.ipynb çalıştırılınca yüklenir).")
        return
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _berturk_tokenizer = AutoTokenizer.from_pretrained(path)
        _berturk_model     = AutoModelForSequenceClassification.from_pretrained(path)
        _berturk_model.to(_berturk_device)
        _berturk_model.eval()
        print("[app] BerTURK modeli yüklendi.")
    except Exception as e:
        print(f"[app] BerTURK yüklenirken hata: {e}")


_load_classic_models()
_load_bilstm()
_load_berturk()


# ─── Yardımcı ─────────────────────────────────────────────────────────────
def _select_classic(model_name: str):
    if model_name in _MODELS:
        return _MODELS[model_name], _MODEL_NAMES[model_name]
    for key in ["svm", "lr", "nb", "rf"]:
        if key in _MODELS:
            return _MODELS[key], _MODEL_NAMES[key]
    return None, None


# ─── Rotalar ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()

    data       = request.get_json(force=True, silent=True) or {}
    raw_text   = data.get("text", "").strip()
    model_name = data.get("model", "svm").lower()

    if not raw_text:
        return jsonify({"error": "Metin boş olamaz."}), 400

    # RegEx temizleme
    t_clean = time.perf_counter()
    cleaned = clean_tweet(raw_text)
    t_clean_ms = (time.perf_counter() - t_clean) * 1000

    if cleaned is None:
        return jsonify({
            "error": "Geçersiz veya yetersiz metin uzunluğu (en az 3, en fazla 100 kelime)."
        }), 400

    features = extract_features(cleaned)

    # Tahmin
    t_infer = time.perf_counter()

    if model_name == "bilstm":
        if _bilstm_model is None:
            return jsonify({
                "error": "BiLSTM modeli henüz yüklenmedi. faz2_bilstm.ipynb çalıştırın."
            }), 503
        pred, confidence = bilstm_predict(
            cleaned, _bilstm_model, _bilstm_vocab, _bilstm_maxlen, _bilstm_device
        )
        is_fake    = pred == 0
        model_used = "BiLSTM"
    elif model_name == "berturk":
        if _berturk_model is None:
            return jsonify({
                "error": "BerTURK modeli henüz yüklenmedi. faz2_berturk.ipynb çalıştırın."
            }), 503
        enc = _berturk_tokenizer(
            cleaned, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(_berturk_device)
        attention_mask = enc["attention_mask"].to(_berturk_device)
        with torch.no_grad():
            logits = _berturk_model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs      = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred       = int(logits.argmax(dim=1).item())
        is_fake    = pred == 0
        confidence = float(probs[0] if is_fake else probs[1])
        model_used = "BerTURK"
    else:
        model, model_used = _select_classic(model_name)
        if model is None:
            return jsonify({"error": "Model bulunamadı. 'python train_model.py' çalıştırın."}), 503
        proba      = model.predict_proba([cleaned])[0]
        pred       = model.predict([cleaned])[0]
        is_fake    = int(pred) == 0
        confidence = float(proba[0] if is_fake else proba[1])

    t_infer_ms = (time.perf_counter() - t_infer) * 1000
    label      = "SAHTE" if is_fake else "GERÇEK"
    latency_ms = (time.perf_counter() - t0) * 1000

    return jsonify({
        "label":        label,
        "confidence":   round(confidence, 4),
        "is_fake":      is_fake,
        "model_used":   model_used,
        "features": {
            "word_count":           features["word_count"],
            "text_length":          features["text_length"],
            "unique_words_ratio":   round(features["unique_words_ratio"], 3),
            "punctuation_ratio":    round(features["punctuation_ratio"], 4),
            "avg_word_length":      round(features["avg_word_length"], 2),
            "sentence_count":       features["sentence_count"],
        },
        "latency_ms":   round(latency_ms, 2),
        "clean_ms":     round(t_clean_ms, 2),
        "infer_ms":     round(t_infer_ms, 2),
        "cleaned_text": cleaned,
    })


@app.route("/health")
def health():
    loaded = list(_MODELS.keys())
    if _bilstm_model is not None:
        loaded.append("bilstm")
    if _berturk_model is not None:
        loaded.append("berturk")
    return jsonify({
        "status":        "ok",
        "loaded_models": loaded,
    })


if __name__ == "__main__":
    app.run(debug=True, port=8080)
