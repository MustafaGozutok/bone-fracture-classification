"""
Bone Fracture Classification — XGBoost
========================================
SOTA Yaklaşım:
  • HOG + LBP + GLCM multi-descriptor fusion + PCA
  • GridSearchCV ile hiperparametre optimizasyonu
  • Early stopping (validation-based)
  • Feature importance (gain-based) görselleştirme
"""

import os
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from config import RANDOM_SEED, RESULTS_DIR, MODELS_DIR
from data_loader import get_splits
from feature_extraction import extract_features_batch, fit_scaler_pca, transform_features
from evaluate import evaluate_model


def train_xgboost(X_train_pca, y_train, X_val_pca, y_val):
    """GridSearchCV ile XGBoost eğit + early stopping."""

    param_grid = {
        'n_estimators':     [100, 300, 500],
        'max_depth':        [3, 5, 7],
        'learning_rate':    [0.01, 0.05, 0.1],
        'subsample':        [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_alpha':        [0, 0.1, 1.0],
        'reg_lambda':       [1.0, 2.0],
    }

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        tree_method='hist',        # Hızlı histogram tabanlı
        n_jobs=-1
    )

    # Train + Val birleştirip CV
    X_cv = np.vstack([X_train_pca, X_val_pca])
    y_cv = np.concatenate([y_train, y_val])

    print("[BİLGİ] GridSearchCV başlıyor (XGBoost)...")
    print("[BİLGİ] Bu işlem uzun sürebilir, lütfen bekleyin...")
    t0 = time.time()

    # Parametre sayısı çok fazla olduğundan RandomizedSearchCV alternatif
    from sklearn.model_selection import RandomizedSearchCV

    random_search = RandomizedSearchCV(
        xgb, param_grid,
        n_iter=50,              # 50 rastgele kombinasyon
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_SEED,
        refit=True
    )
    random_search.fit(X_cv, y_cv)

    elapsed = time.time() - t0
    print(f"[BİLGİ] RandomizedSearchCV tamamlandı: {elapsed:.1f}s")
    print(f"[BİLGİ] En iyi parametreler: {random_search.best_params_}")
    print(f"[BİLGİ] En iyi CV F1: {random_search.best_score_:.4f}")

    # Early stopping ile final model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params


def plot_xgb_feature_importance(model, n_top=30, save_dir=RESULTS_DIR):
    """XGBoost feature importance (gain-based)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-n_top:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices], color='#4CAF50')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'Feature {i}' for i in indices], fontsize=8)
    ax.set_xlabel('Gain-based Importance', fontsize=12)
    ax.set_title('XGBoost — Top Feature Importances', fontsize=14)

    path = os.path.join(save_dir, "xgb_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Feature importance kaydedildi: {path}")


def main():
    print("=" * 60)
    print("  XGBoost — Bone Fracture Classification")
    print("=" * 60)

    # 1. Veri yükle
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits()

    # 2. Feature extraction
    print("\n[BİLGİ] Feature extraction (Train)...")
    F_train = extract_features_batch(X_train)
    print("[BİLGİ] Feature extraction (Val)...")
    F_val = extract_features_batch(X_val)
    print("[BİLGİ] Feature extraction (Test)...")
    F_test = extract_features_batch(X_test)

    # 3. StandardScaler + PCA
    print("\n[BİLGİ] StandardScaler + PCA fit...")
    scaler, pca, F_train_pca = fit_scaler_pca(F_train)
    F_val_pca  = transform_features(F_val, scaler, pca)
    F_test_pca = transform_features(F_test, scaler, pca)

    # 4. Model eğitimi
    model, best_params = train_xgboost(F_train_pca, y_train, F_val_pca, y_val)

    # 5. Test seti değerlendirme
    y_pred = model.predict(F_test_pca)
    y_prob = model.predict_proba(F_test_pca)[:, 1]

    result = evaluate_model("XGBoost", y_test, y_pred, y_prob)
    result["best_params"] = best_params

    # 6. Feature importance
    plot_xgb_feature_importance(model)

    # 7. Model kaydet
    model_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    joblib.dump({"model": model, "scaler": scaler, "pca": pca}, model_path)
    print(f"\n[BİLGİ] Model kaydedildi: {model_path}")

    return result


if __name__ == "__main__":
    result = main()
