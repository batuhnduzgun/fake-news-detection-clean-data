"""
bilstm_model.py — BiLSTM model sınıfı ve tahmin yardımcıları.
Hem faz2_bilstm.ipynb hem de app.py tarafından kullanılır.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Rapor Bölüm 5.1.1:
      - Embedding katmanı
      - 128 birimlik çift yönlü LSTM
      - Dropout 0.2
      - Sigmoid Dense çıkış katmanı
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, _) = self.bilstm(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        h = self.dropout(h)
        logit = self.fc(h).squeeze(1)
        return torch.sigmoid(logit)


def load_bilstm(model_path: str, device=None):
    """
    Kaydedilmiş BiLSTM modelini yükler.
    Döndürür: (model, word2idx, max_len) veya None
    """
    import os
    if not os.path.exists(model_path):
        return None, None, None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    word2idx = checkpoint["vocab"]
    max_len  = checkpoint["max_len"]

    model = BiLSTMClassifier(len(word2idx))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, word2idx, max_len


def bilstm_predict(text: str, model, word2idx: dict, max_len: int, device=None):
    """
    Tek bir metni BiLSTM ile tahmin eder.
    Döndürür: (label: int, confidence: float)
      label 0 = sahte, 1 = gerçek
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    UNK = 1
    PAD = 0
    tokens = text.split()[:max_len]
    ids = [word2idx.get(t, UNK) for t in tokens]
    ids += [PAD] * (max_len - len(ids))

    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        prob = model(x).item()

    label = 1 if prob >= 0.5 else 0
    confidence = prob if label == 1 else 1 - prob
    return label, confidence
