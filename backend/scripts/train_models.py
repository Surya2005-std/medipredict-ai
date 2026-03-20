"""
train_models.py
────────────────────────────────────────────────────────────────
Trains multiple ML classifiers on the disease-symptom dataset,
compares performance, selects the best model, and saves everything
needed for inference.

Models trained:
  • Random Forest
  • Gradient Boosting
  • Support Vector Machine (RBF kernel)
  • Naive Bayes (Gaussian)

Outputs (saved to backend/models/):
  • best_model.pkl       – the winning classifier
  • label_encoder.pkl    – LabelEncoder for disease names
  • feature_names.pkl    – ordered symptom feature list
  • model_metrics.json   – accuracy/precision/recall/F1 for all models
  • confusion_matrix.png – confusion matrix of the best model
  • feature_importance.png – top-30 feature importances (if applicable)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_PATH  = os.path.join(DATA_DIR, 'disease_symptoms.csv')


# ══════════════════════════════════════════════════════════════
# 1.  LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════
def load_and_preprocess():
    print("\n📂 Loading dataset …")
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    target_col = 'Disease'
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].fillna(0).astype(int)
    y_raw = df[target_col].str.strip()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"   ✔ Dataset: {X.shape[0]} samples  |  {X.shape[1]} features  |  {len(le.classes_)} classes")
    print(f"   ✔ Diseases: {', '.join(le.classes_[:5])} … (and {len(le.classes_)-5} more)")

    return X, y, le, feature_cols


# ══════════════════════════════════════════════════════════════
# 2.  FEATURE SELECTION
# ══════════════════════════════════════════════════════════════
def select_features(X, y, feature_cols, threshold='mean'):
    print("\n🔍 Running feature selection with Random Forest …")
    selector_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector_rf.fit(X, y)

    selector = SelectFromModel(selector_rf, threshold=threshold, prefit=True)
    X_sel = selector.transform(X)
    selected_mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_cols, selected_mask) if m]

    print(f"   ✔ Features selected: {X_sel.shape[1]} / {X.shape[1]}")
    return X_sel, selected_features, selector_rf


# ══════════════════════════════════════════════════════════════
# 3.  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════
def get_models():
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_split=2,
            random_state=42
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=50,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        'Naive Bayes': GaussianNB(
            var_smoothing=1e-9
        ),
    }


# ══════════════════════════════════════════════════════════════
# 4.  TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════
def train_and_evaluate(X, y, le):
    print("\n🏋️  Training and evaluating models …")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models  = get_models()
    results = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n   ▶ {name}")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring='accuracy', n_jobs=-1)

        # Full train + test evaluation
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {
            'model':      model,
            'accuracy':   round(acc,  4),
            'precision':  round(prec, 4),
            'recall':     round(rec,  4),
            'f1':         round(f1,   4),
            'cv_mean':    round(cv_scores.mean(), 4),
            'cv_std':     round(cv_scores.std(),  4),
            'y_pred':     y_pred,
        }

        print(f"     CV accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"     Test accuracy: {acc:.4f}  |  F1: {f1:.4f}")

    return results, X_test, y_test


# ══════════════════════════════════════════════════════════════
# 5.  SELECT BEST MODEL
# ══════════════════════════════════════════════════════════════
def pick_best(results):
    best_name = max(results, key=lambda k: results[k]['f1'])
    best      = results[best_name]
    print(f"\n🏆 Best model: {best_name}  (F1={best['f1']:.4f}  Acc={best['accuracy']:.4f})")
    return best_name, best['model']


# ══════════════════════════════════════════════════════════════
# 6.  SAVE ARTEFACTS
# ══════════════════════════════════════════════════════════════
def save_artefacts(best_name, best_model, le, selected_features, results):
    joblib.dump(best_model,        os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(le,                os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    joblib.dump(selected_features, os.path.join(MODELS_DIR, 'feature_names.pkl'))

    # Metrics JSON (without model objects)
    metrics = {}
    for name, v in results.items():
        metrics[name] = {k: v[k] for k in ('accuracy','precision','recall','f1','cv_mean','cv_std')}
    metrics['best_model'] = best_name

    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Artefacts saved to {MODELS_DIR}")


# ══════════════════════════════════════════════════════════════
# 7.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════
def plot_confusion_matrix(best_name, best_model, X_test, y_test, le):
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax, linewidths=0.5, annot_kws={"size": 7})
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual',    fontsize=13)
    ax.set_title(f'Confusion Matrix – {best_name}', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✔ Confusion matrix saved → {path}")


def plot_feature_importance(best_name, best_model, selected_features, top_n=30):
    if not hasattr(best_model, 'feature_importances_'):
        print("   ⚠  Feature importance not available for this model type.")
        return

    importances = best_model.feature_importances_
    fi_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1],
                   color=plt.cm.viridis(np.linspace(0.2, 0.85, top_n)))
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances – {best_name}', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✔ Feature importance saved → {path}")


def plot_model_comparison(results):
    names   = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors  = ['#4361EE', '#3A86FF', '#7B2D8B', '#E63946']

    x   = np.arange(len(names))
    w   = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(x + i * w, vals, w, label=metric.capitalize(), color=color, alpha=0.85)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison – Accuracy / Precision / Recall / F1', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left','bottom']:
        ax.spines[spine].set_color('#ccc')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✔ Model comparison saved → {path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("   DISEASE SYMPTOM PREDICTOR – MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    X_full, y, le, feature_cols = load_and_preprocess()

    # 2. Use ALL features for maximum accuracy (selection was informational)
    # We keep full feature list so the predictor always has all 132 symptom columns
    X_sel, selected_features = X_full, feature_cols
    print(f"\n✅ Using ALL {len(selected_features)} features for maximum accuracy")

    # 3. Train & evaluate
    results, X_test, y_test = train_and_evaluate(X_sel, y, le)

    # 4. Pick best
    best_name, best_model = pick_best(results)

    # 5. Save artefacts
    save_artefacts(best_name, best_model, le, selected_features, results)

    # 6. Plots
    print("\n📊 Generating visualisations …")
    plot_confusion_matrix(best_name, best_model, X_test, y_test, le)
    plot_feature_importance(best_name, best_model, selected_features)
    plot_model_comparison(results)

    # 7. Summary
    print("\n" + "=" * 60)
    print("   TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Best Model : {best_name}")
    r = results[best_name]
    print(f"  Accuracy   : {r['accuracy']*100:.2f}%")
    print(f"  Precision  : {r['precision']*100:.2f}%")
    print(f"  Recall     : {r['recall']*100:.2f}%")
    print(f"  F1-Score   : {r['f1']*100:.2f}%")
    print(f"  CV Mean    : {r['cv_mean']*100:.2f}% ± {r['cv_std']*100:.2f}%")
    print()


if __name__ == '__main__':
    main()
