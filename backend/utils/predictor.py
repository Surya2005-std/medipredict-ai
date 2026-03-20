"""
predictor.py  –  Inference module for MediPredict AI
──────────────────────────────────────────────────────
All 132 symptoms are now valid inputs. The model was trained
on all features so every symptom affects predictions.
"""

import os, json
import numpy as np
import pandas as pd
import joblib

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

_model    = None
_encoder  = None
_features = None
_prec_df  = None
_sev_map  = None
_metrics  = None


def _load():
    global _model, _encoder, _features, _prec_df, _sev_map, _metrics
    if _model is not None:
        return

    _model    = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    _encoder  = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    _features = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))

    prec_path = os.path.join(DATA_DIR, 'disease_precautions.csv')
    _prec_df  = pd.read_csv(prec_path) if os.path.exists(prec_path) else pd.DataFrame()

    sev_path = os.path.join(DATA_DIR, 'symptom_severity.csv')
    if os.path.exists(sev_path):
        sev_df   = pd.read_csv(sev_path)
        _sev_map = dict(zip(sev_df['Symptom'], sev_df['Severity']))
    else:
        _sev_map = {}

    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _metrics = json.load(f)
    else:
        _metrics = {}


def get_all_symptoms() -> list:
    """Return all 132 symptom names for UI search/autocomplete."""
    _load()
    return list(_features)


def get_metrics() -> dict:
    _load()
    return _metrics or {}


def _get_precautions(disease: str) -> list:
    if _prec_df is None or _prec_df.empty:
        return []
    row = _prec_df[_prec_df['Disease'] == disease]
    if row.empty:
        return []
    cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    return [str(row.iloc[0][c]).strip()
            for c in cols if c in row.columns
            and str(row.iloc[0][c]).strip() not in ('', 'nan')]


def predict(symptoms: list, top_k: int = 3) -> dict:
    _load()

    # Normalise input
    clean = [s.strip().lower().replace(' ', '_') for s in symptoms]

    # Build feature vector
    vec = np.zeros(len(_features), dtype=int)
    matched, unknown = [], []
    for sym in clean:
        if sym in _features:
            vec[_features.index(sym)] = 1
            matched.append(sym)
        else:
            unknown.append(sym)

    if not matched:
        return {
            'error': 'No recognisable symptoms provided. Please select symptoms from the dropdown list.',
            'predictions': [],
            'matched_symptoms': [],
            'unknown_symptoms': clean,
            'severity_score': 0,
            'model_info': {},
        }

    # Predict
    proba       = _model.predict_proba(vec.reshape(1, -1))[0]
    top_indices = np.argsort(proba)[::-1][:top_k]

    predictions = []
    for idx in top_indices:
        disease    = _encoder.inverse_transform([idx])[0]
        confidence = round(float(proba[idx]) * 100, 2)
        predictions.append({
            'disease':     disease,
            'probability': confidence,
            'precautions': _get_precautions(disease),
        })

    # Severity score
    sev_vals       = [_sev_map.get(s, 3) for s in matched]
    severity_score = round(min(np.mean(sev_vals) / 7 * 100, 100), 1) if sev_vals else 0

    best_name = _metrics.get('best_model', 'Unknown')
    model_acc = _metrics.get(best_name, {}).get('accuracy', 0)

    return {
        'predictions':      predictions,
        'matched_symptoms': matched,
        'unknown_symptoms': unknown,
        'severity_score':   severity_score,
        'model_info':       {'name': best_name, 'accuracy': round(model_acc * 100, 2)},
    }
