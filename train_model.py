"""
train_model.py — Tüm 4 klasik modeli eğit ve diske kaydet.

Web arayüzünün (app.py) kullanacağı model ağırlıklarını üretir.
Çalıştırmak için:
    python train_model.py
"""

import os
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_processing import prepare_balanced_data


def _make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", clf),
    ])


def train_and_save(
    real_news_path: str = "data/real-news.txt",
    fake_news_path: str = "data/fake-news.txt",
    output_dir: str = "models",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("Veri yükleniyor...")
    train_df, test_df = prepare_balanced_data(real_news_path, fake_news_path)
    print(f"  Eğitim: {len(train_df)} örnek | Test: {len(test_df)} örnek\n")

    # LinearSVC + CalibratedClassifierCV: SVC(kernel='linear')'den ~100x hızlı,
    # aynı zamanda predict_proba destekler.
    svm_clf = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42))

    models = {
        "nb":  ("Naive Bayes",         _make_pipeline(MultinomialNB())),
        "rf":  ("Random Forest",        _make_pipeline(RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))),
        "lr":  ("Lojistik Regresyon",   _make_pipeline(LogisticRegression(l1_ratio=0, random_state=42, max_iter=1000))),
        "svm": ("SVM",                  _make_pipeline(svm_clf)),
    }

    for key, (name, pipeline) in models.items():
        print(f"{name} eğitiliyor...")
        pipeline.fit(train_df["text"], train_df["is_real"])
        preds = pipeline.predict(test_df["text"])
        acc = accuracy_score(test_df["is_real"], preds)
        print(f"  Doğruluk: {acc:.4f}")
        path = os.path.join(output_dir, f"{key}_pipeline.joblib")
        joblib.dump(pipeline, path)
        print(f"  Kaydedildi → {path}\n")

    print("Tüm modeller kaydedildi.")


if __name__ == "__main__":
    train_and_save()
