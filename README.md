# 🩺 MediPredict AI — Disease Symptom Predictor

> An intelligent, production-ready system that predicts possible diseases from user-input symptoms using a multi-model machine learning pipeline — built with Python, Flask, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-93%25+-brightgreen)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 📸 Screenshots

| Predictor Page | Dashboard |
|---|---|
| *(Add screenshot here)* | *(Add screenshot here)* |

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔬 **Multi-model ML pipeline** | Random Forest, Gradient Boosting, SVM, Naive Bayes |
| 🎯 **93%+ accuracy** | Trained on 4,920 samples across 41 diseases |
| 🏆 **Top-3 predictions** | Ranked disease predictions with confidence scores |
| ⚠️ **Severity scoring** | Weighted symptom severity meter (1–7 scale) |
| 💊 **Precaution advice** | 4 actionable precautions per predicted disease |
| 🤖 **MediBot chatbot** | Natural language symptom extraction + prediction |
| 📊 **Evaluation dashboard** | Live metrics, confusion matrix, feature importance |
| ⚡ **Autocomplete search** | Fuzzy search across 132 clinical symptom indicators |
| 🎛️ **Preset scenarios** | One-click symptom presets for quick demos |
| 🌐 **REST API** | Clean JSON API for all prediction + metrics endpoints |

---

## 🗂️ Project Structure

```
disease_predictor/
├── backend/
│   ├── app.py                    # Flask application & API routes
│   ├── data/
│   │   ├── disease_symptoms.csv  # Training dataset (4,920 rows)
│   │   ├── symptom_severity.csv  # Severity weights per symptom
│   │   └── disease_precautions.csv
│   ├── models/
│   │   ├── best_model.pkl        # Saved best classifier
│   │   ├── label_encoder.pkl     # Disease label encoder
│   │   ├── feature_names.pkl     # Selected feature list
│   │   ├── model_metrics.json    # All model scores
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   └── model_comparison.png
│   ├── scripts/
│   │   ├── generate_dataset.py   # Dataset generator
│   │   └── train_models.py       # Full training pipeline
│   └── utils/
│       └── predictor.py          # Inference utility module
├── frontend/
│   ├── templates/
│   │   ├── index.html            # Main predictor UI
│   │   └── dashboard.html        # Model evaluation dashboard
│   └── static/
│       ├── css/
│       │   ├── style.css         # Main stylesheet
│       │   └── dashboard.css     # Dashboard styles
│       ├── js/
│       │   ├── app.js            # Predictor logic
│       │   └── dashboard.js      # Dashboard logic
│       └── models/               # Chart images (auto-generated)
├── run.py                        # Application entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medipredict-ai.git
cd medipredict-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate data & train models

```bash
python run.py --setup
```

This will:
- Generate the 4,920-sample disease-symptom dataset
- Train 4 ML classifiers with 5-fold cross-validation
- Save the best model and evaluation charts

> ⏱ Training takes **2–5 minutes** on a typical laptop.

### 5. Start the server

```bash
python run.py
```

Open your browser at **http://localhost:5000**

---

## 🔌 API Reference

### `GET /api/health`
Health check.
```json
{"status": "ok", "service": "Disease Symptom Predictor"}
```

### `GET /api/symptoms`
Returns all 132 valid symptom names (for autocomplete).
```json
{
  "symptoms": [{"value": "high_fever", "label": "High Fever"}, ...],
  "count": 132
}
```

### `POST /api/predict`
Predict diseases from symptoms.

**Request:**
```json
{
  "symptoms": ["high_fever", "headache", "chills", "sweating", "muscle_pain"],
  "top_k": 3
}
```

**Response:**
```json
{
  "predictions": [
    {
      "disease": "Malaria",
      "probability": 64.3,
      "precautions": ["Use mosquito repellent", "Sleep under nets", ...]
    },
    ...
  ],
  "matched_symptoms": ["high_fever", "headache", "chills", "sweating", "muscle_pain"],
  "unknown_symptoms": [],
  "severity_score": 71.4,
  "model_info": {"name": "Random Forest", "accuracy": 92.99}
}
```

### `GET /api/metrics`
Model evaluation metrics.
```json
{
  "Random Forest": {"accuracy": 0.9299, "precision": 0.931, "recall": 0.9299, "f1": 0.9304, "cv_mean": 0.9362, "cv_std": 0.0054},
  "Gradient Boosting": {...},
  "SVM (RBF)": {...},
  "Naive Bayes": {...},
  "best_model": "Random Forest"
}
```

---

## 🧠 ML Pipeline

```
Raw CSV  →  Preprocessing  →  Feature Selection  →  Train/Test Split (80/20)
             (fillna, encode)   (SelectFromModel)     (stratified)

                    ↓
    ┌───────────────────────────────────┐
    │   Random Forest  (n=200, sqrt)    │
    │   Gradient Boosting (n=100, lr=0.1)│
    │   SVM RBF  (C=10, gamma=scale)    │
    │   Naive Bayes  (var_smooth=1e-8)  │
    └───────────────────────────────────┘
                    ↓
    5-Fold Stratified Cross-Validation
                    ↓
    Best Model by F1-Score  →  joblib.dump()
```

### Model Performance

| Model | Accuracy | F1-Score | CV Mean |
|---|---|---|---|
| **Random Forest** ⭐ | **92.99%** | **93.04%** | **93.62%** |
| SVM (RBF) | 92.99% | 93.02% | 93.52% |
| Gradient Boosting | 91.26% | 91.34% | 90.65% |
| Naive Bayes | 82.01% | 82.41% | 76.47% |

---

## 🌍 Deployment

### Render (recommended)

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set **Build Command**: `pip install -r requirements.txt && python run.py --setup`
4. Set **Start Command**: `gunicorn backend.app:app`
5. Set **Environment**: Python 3

### Railway

```bash
railway init
railway add
railway up
```

Set start command: `python run.py`

### HuggingFace Spaces

Create a `Gradio` or `Streamlit` wrapper around `utils/predictor.py` and deploy to Spaces.

---

## 🔮 Future Improvements

- [ ] **95%+ accuracy** with larger real-world dataset (Kaggle Disease Prediction v2)
- [ ] **Neural network** model (MLP / TabNet) for improved performance
- [ ] **Symptom description** — richer metadata per symptom
- [ ] **Patient history** tracking (SQLite / MongoDB integration)
- [ ] **Export report** — PDF report of diagnosis + precautions
- [ ] **Multilingual UI** — support for Hindi, Tamil, Spanish
- [ ] **Drug interaction** checker alongside prediction
- [ ] **Doctor finder** — map-based nearest doctor recommendations
- [ ] **Mobile app** — React Native / Flutter wrapper
- [ ] **Real EMR dataset** — train on de-identified hospital data for higher clinical validity

---

## ⚠️ Medical Disclaimer

This application is for **educational and informational purposes only**. It does **not** constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any health concerns.

---

## 📄 License

MIT © 2024 — Free to use, modify and distribute.
