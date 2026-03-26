"""
Bone Fracture Classification — Decision Tree
==============================================
SOTA Yaklaşım:
  • HOG + LBP + GLCM multi-descriptor fusion
  • PCA boyut azaltma (%95 varyans)
  • GridSearchCV ile hiperparametre optimizasyonu
  • Feature Importance analizi
"""

import os
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from config import RANDOM_SEED, RESULTS_DIR, MODELS_DIR
from data_loader import get_splits
from feature_extraction import extract_features_batch, fit_scaler_pca, transform_features
from evaluate import evaluate_model


def train_decision_tree(X_train_pca, y_train, X_val_pca, y_val):
    """GridSearchCV ile Decision Tree eğit."""

    param_grid = {
        'criterion':        ['gini', 'entropy'],
        'max_depth':        [5, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf':  [1, 2, 5, 10],
        'max_features':     ['sqrt', 'log2', None],
    }

    dt = DecisionTreeClassifier(random_state=RANDOM_SEED)

    # Train + Val birleştirip cross-validation
    X_cv = np.vstack([X_train_pca, X_val_pca])
    y_cv = np.concatenate([y_train, y_val])

    print("[BİLGİ] GridSearchCV başlıyor (Decision Tree)...")
    t0 = time.time()

    grid_search = GridSearchCV(
        dt, param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_cv, y_cv)

    elapsed = time.time() - t0
    print(f"[BİLGİ] GridSearchCV tamamlandı: {elapsed:.1f}s")
    print(f"[BİLGİ] En iyi parametreler: {grid_search.best_params_}")
    print(f"[BİLGİ] En iyi CV F1: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def plot_feature_importance(model, n_top=30, save_dir=RESULTS_DIR):
    """En önemli N özelliğin bar chart'ı."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-n_top:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices], color='#2196F3')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f'Feature {i}' for i in indices], fontsize=8)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Decision Tree — Top Feature Importances', fontsize=14)

    path = os.path.join(save_dir, "dt_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Feature importance kaydedildi: {path}")


def main():
    print("=" * 60)
    print("  DECISION TREE — Bone Fracture Classification")
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
    model, best_params = train_decision_tree(F_train_pca, y_train, F_val_pca, y_val)

    # 5. Test seti değerlendirme
    y_pred = model.predict(F_test_pca)
    y_prob = model.predict_proba(F_test_pca)[:, 1]

    result = evaluate_model("Decision Tree", y_test, y_pred, y_prob)
    result["best_params"] = best_params

    # 6. Feature importance
    plot_feature_importance(model)

    # 7. Model kaydet
    model_path = os.path.join(MODELS_DIR, "decision_tree.pkl")
    joblib.dump({"model": model, "scaler": scaler, "pca": pca}, model_path)
    print(f"\n[BİLGİ] Model kaydedildi: {model_path}")

    return result


if __name__ == "__main__":
    result = main()
