"""
Bone Fracture Classification - Naive Bayes (GaussianNB)
======================================================
Classical pipeline:
  - HOG + LBP + GLCM feature fusion + PCA
  - GridSearchCV for var_smoothing
"""

import os
import time
import joblib
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

from config import MODELS_DIR
from data_loader import get_splits
from feature_extraction import extract_features_batch, fit_scaler_pca, transform_features
from evaluate import evaluate_model


def train_naive_bayes(X_train_pca, y_train, X_val_pca, y_val):
    """Train GaussianNB with GridSearchCV."""

    param_grid = {
        "var_smoothing": np.logspace(-12, -6, 13)
    }

    nb = GaussianNB()

    # Merge train + val for CV search
    X_cv = np.vstack([X_train_pca, X_val_pca])
    y_cv = np.concatenate([y_train, y_val])

    print("[INFO] GridSearchCV starting (Naive Bayes)...")
    t0 = time.time()

    grid_search = GridSearchCV(
        nb,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X_cv, y_cv)

    elapsed = time.time() - t0
    print(f"[INFO] GridSearchCV finished: {elapsed:.1f}s")
    print(f"[INFO] Best params: {grid_search.best_params_}")
    print(f"[INFO] Best CV F1: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def main():
    print("=" * 60)
    print("  NAIVE BAYES - Bone Fracture Classification")
    print("=" * 60)

    # 1. Load splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits()

    # 2. Feature extraction
    print("\n[INFO] Feature extraction (Train)...")
    F_train = extract_features_batch(X_train)
    print("[INFO] Feature extraction (Val)...")
    F_val = extract_features_batch(X_val)
    print("[INFO] Feature extraction (Test)...")
    F_test = extract_features_batch(X_test)

    # 3. StandardScaler + PCA
    print("\n[INFO] StandardScaler + PCA fit...")
    scaler, pca, F_train_pca = fit_scaler_pca(F_train)
    F_val_pca = transform_features(F_val, scaler, pca)
    F_test_pca = transform_features(F_test, scaler, pca)

    # 4. Train model
    model, best_params = train_naive_bayes(F_train_pca, y_train, F_val_pca, y_val)

    # 5. Evaluate on test
    y_pred = model.predict(F_test_pca)
    y_prob = model.predict_proba(F_test_pca)[:, 1]

    result = evaluate_model("Naive Bayes", y_test, y_pred, y_prob)
    result["best_params"] = best_params

    # 6. Save model
    model_path = os.path.join(MODELS_DIR, "naive_bayes.pkl")
    joblib.dump({"model": model, "scaler": scaler, "pca": pca}, model_path)
    print(f"\n[INFO] Model saved: {model_path}")

    return result


if __name__ == "__main__":
    result = main()
