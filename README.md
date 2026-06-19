# Yapay Zeka Destekli Türkçe Sahte Haber ve Dezenformasyon Tespit Sistemi

**Batuhan DÜZGÜN — 2210656034**  
Tekirdağ Namık Kemal Üniversitesi · Çorlu Mühendislik Fakültesi  
Danışman: Prof. Dr. Erdinç UZUN · Haziran 2026

---

## İçindekiler

1. [Proje Yapısı](#proje-yapısı)
2. [Kurulum](#kurulum)
3. [Adım 1 — Veriyi Hazırla](#adım-1--veriyi-hazırla)
4. [Adım 2 — Faz 1: Klasik ML Analizleri](#adım-2--faz-1-klasik-ml-analizleri)
5. [Adım 3 — Faz 2: Derin Öğrenme Modelleri](#adım-3--faz-2-derin-öğrenme-modelleri)
6. [Adım 4 — Web Arayüzünü Çalıştır](#adım-4--web-arayüzünü-çalıştır)
7. [Model Performans Özeti](#model-performans-özeti)
8. [Dosya Referansı](#dosya-referansı)

---

## Proje Yapısı

```
fake-news-detection-clean-data-main/
│
├── data/                        ← Temizlenmiş veri dosyaları
│   ├── real-news.txt            ← 17.452 gerçek haber
│   └── fake-news.txt            ← 13.447 sahte haber
│
├── raw-data/                    ← Ham kaynak veriler
│   ├── fake/                    ← DMM, Community Notes, Doğruluk Payı
│   └── real/                    ← Ana akım haber ajansları (JSON)
│
├── ── NOTEBOOK'LAR ──────────────────────────────────────
├── balanced.ipynb               ← Faz 1 — Dengeli senaryo
├── imbalanced_fake.ipynb        ← Faz 1 — Dengesiz senaryo (2:1)
├── constant_fake.ipynb          ← Faz 1 — Sabit sahte senaryo
├── t.ipynb                      ← LSTM referans + Faz1 vs Faz2 karşılaştırma
├── faz2_bilstm.ipynb            ← Faz 2 — BiLSTM (PyTorch)
├── faz2_berturk.ipynb           ← Faz 2 — BerTURK (Transformer)
│
├── ── WEB UYGULAMASI ────────────────────────────────────
├── train_model.py               ← Modeli eğit ve kaydet
├── app.py                       ← Flask backend API
├── templates/index.html         ← Web arayüzü
├── static/style.css             ← Stil dosyası
├── models/                      ← Eğitilmiş model dosyaları (train sonrası oluşur)
│   ├── svm_pipeline.joblib
│   └── lr_pipeline.joblib
│
├── ── YARDIMCI MODÜLLER ─────────────────────────────────
├── cleaner.py                   ← RegEx temizleme + öznitelik çıkarımı
├── data_processing.py           ← Veri yükleme ve senaryo hazırlama
├── ml_utils.py                  ← Model eğitim + değerlendirme pipeline
├── graphic.py                   ← Tüm görselleştirme fonksiyonları
└── turkish_stop_words.py        ← 54 Türkçe durak kelime listesi
```

---

## Kurulum

### Gereksinimler

- Python 3.10 veya üzeri
- pip

### Paketleri Yükle

```bash
pip install scikit-learn pandas numpy plotly flask flask-cors joblib torch transformers
```

---

## Adım 1 — Veriyi Hazırla

> Ham veriler `raw-data/` klasöründe zaten mevcut.  
> `data/` klasöründeki temizlenmiş dosyalar da hazır.  
> Ham veriden yeniden temizleme yapmak istersen:

```bash
python cleaner.py
```

Bu komut `raw-data/` altındaki tüm ham haberleri okur, temizler ve `data/real-news.txt` ile `data/fake-news.txt` dosyalarını oluşturur.

---

## Adım 2 — Faz 1: Klasik ML Analizleri

Üç farklı veri dağılım senaryosu için ayrı notebook'lar bulunmaktadır.  
Her birini Jupyter'de aç ve tüm hücreleri sırayla çalıştır:

### 2a. Dengeli Senaryo — `balanced.ipynb`

```
Jupyter'de aç → Kernel → Restart & Run All
```

- Her sınıftan eşit sayıda veri (~7.317 eğitim, ~1.829 test)
- 4 model test edilir: **NB, RF, LR, SVM**
- Her model için: ROC eğrisi, öğrenme eğrisi, sınıflandırma raporu
- Son hücre: SVM ve LR için yan yana **Karmaşıklık Matrisi** (Şekil 1)

### 2b. Dengesiz Senaryo — `imbalanced_fake.ipynb`

```
Jupyter'de aç → Kernel → Restart & Run All
```

- Gerçek : Sahte oranı = **2:1** (gerçek hayat simülasyonu)
- Aynı 4 model, dengesizliğe karşı direnç testi
- Son hücre: Karmaşıklık Matrisi

### 2c. Sabit Sahte Senaryo — `constant_fake.ipynb`

```
Jupyter'de aç → Kernel → Restart & Run All
```

- Tüm sahte haberler + 2 kat gerçek haber
- En zorlu asimetri testi

### 2d. LSTM Referans + Faz Karşılaştırma — `t.ipynb`

```
Jupyter'de aç → Kernel → Restart & Run All
```

- LSTM referans model sonuçları (Bölüm 3.5)
- Faz 1 vs Faz 2 karşılaştırmalı çubuk grafikleri

---

## Adım 3 — Faz 2: Derin Öğrenme Modelleri

> ⚠️ Bu notebook'lar uzun süre çalışabilir (CPU: ~30–60 dk, GPU: ~5–10 dk).

### 3a. BiLSTM — `faz2_bilstm.ipynb`

```
Jupyter'de aç → Kernel → Restart & Run All
```

**Mimari (Bölüm 5.1.1):**
- Embedding katmanı
- 128 birimlik çift yönlü LSTM
- Dropout 0.2
- Sigmoid Dense çıkış katmanı
- Adam optimizer, 10 epoch

Çıktılar: öğrenme eğrisi, ROC eğrisi, confusion matrix  
Model dosyası: `models/bilstm_model.pt`

### 3b. BerTURK — `faz2_berturk.ipynb`

```
Jupyter'de aç → Kernel → Restart & Run All
```

> ⚠️ İlk çalıştırmada HuggingFace'den model indirilir (~450 MB). İnternet bağlantısı gereklidir.

**Model:** `dbmdz/bert-base-turkish-128k-uncased`  
**İnce Ayar (Bölüm 5.1.2):**
- Öğrenme oranı: 2×10⁻⁵
- Weight decay: 0.01
- Linear warmup scheduler
- 4 epoch

Çıktılar: öğrenme eğrisi, ROC eğrisi, confusion matrix, Faz karşılaştırma tablosu  
Model dosyası: `models/berturk/`

---

## Adım 4 — Web Arayüzünü Çalıştır

Bu adım, raporun **Bölüm 4**'ünde açıklanan interaktif kullanıcı arayüzünü ayağa kaldırır.

### 4a. Modeli Eğit (bir kez yapılır)

```bash
python train_model.py
```

Bu komut:
- Dengeli veri setiyle SVM ve LR modellerini eğitir (~2 dakika)
- `models/svm_pipeline.joblib` ve `models/lr_pipeline.joblib` dosyalarını oluşturur

> ✅ `models/` klasörü zaten doluysa bu adımı atlayabilirsin.

### 4b. Sunucuyu Başlat

```bash
python app.py
```

Terminalde şunu görürsün:

```
[app] SVM modeli yüklendi.
[app] Lojistik Regresyon modeli yüklendi.
 * Running on http://127.0.0.1:5000
```

### 4c. Tarayıcıda Aç

```
http://localhost:5000
```

### 4d. Nasıl Kullanılır?

1. **Metin Kutusuna** şüpheli haberi yapıştır (tweet, başlık, haber kesiti)
2. **Model seç:** SVM (önerilen) veya Lojistik Regresyon
3. **"Analiz Et"** butonuna tıkla (veya `Ctrl + Enter`)
4. Sonuçlar anında görünür:
   - 🚨 **SAHTE** (kırmızı) veya ✅ **GERÇEK** (yeşil) kararı
   - **Güven Skoru** — modelin karar kesinliğini gösteren yüzde çubuğu
   - **Öznitelik Detayları** — kelime sayısı, cümle sayısı, noktalama oranı vb.
   - **Gecikme Bilgisi** — temizleme ve çıkarım süreleri (milisaniye)

### API Doğrudan Kullanımı

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Haber metni buraya", "model": "svm"}'
```

**Yanıt:**
```json
{
  "label": "GERÇEK",
  "confidence": 0.8234,
  "is_fake": false,
  "model_used": "SVM",
  "features": {
    "word_count": 12,
    "text_length": 78,
    "unique_words_ratio": 0.917,
    "punctuation_ratio": 0.0128,
    "avg_word_length": 5.25,
    "sentence_count": 2
  },
  "latency_ms": 3.47,
  "clean_ms": 0.21,
  "infer_ms": 2.85
}
```

---

## Model Performans Özeti

| Model | Senaryo | Doğruluk | F1-Skoru | AUC |
|---|---|---|---|---|
| **SVM** | **Dengeli** | **%82.5** | **0.825** | **0.91** |
| **Naive Bayes** | **Dengeli** | **%82.5** | **0.825** | **0.91** |
| Lojistik Regresyon | Dengeli | %81.8 | 0.818 | 0.90 |
| Random Forest | Dengeli | %80.7 | 0.807 | 0.89 |
| LSTM (ref) | Dengeli | %77.1 | 0.821 | — |
| BiLSTM | Dengeli | %81.2 | — | — |
| **BerTURK** | **Dengeli** | **%88.1** | — | — |

---

## Dosya Referansı

| Dosya / Fonksiyon | Ne yapar? |
|---|---|
| `cleaner.clean_tweet(text)` | Türkçe regex temizleme, <10ms |
| `cleaner.extract_features(text)` | Kelime sayısı, noktalama vb. öznitelik çıkarır |
| `data_processing.prepare_balanced_data()` | Dengeli senaryo verisi |
| `data_processing.prepare_imbalanced_fake_news_data()` | 2:1 dengesiz senaryo |
| `data_processing.prepare_constant_fake_data()` | Sabit sahte senaryo |
| `ml_utils.train_and_evaluate()` | TF-IDF + model eğitimi + 5-fold CV + ROC + öğrenme eğrisi |
| `graphic.plot_confusion_matrix()` | Tek model karmaşıklık matrisi |
| `graphic.plot_dual_confusion_matrix()` | SVM ve LR yan yana (Şekil 1) |
| `graphic.plot_classification_report()` | Precision/Recall/F1 karşılaştırma grafikleri |
| `graphic.plot_roc_curve_plotly()` | ROC-AUC eğrisi (Şekil 2) |
| `graphic.plot_learning_curves_plotly()` | Öğrenme eğrisi (Şekil 3) |
