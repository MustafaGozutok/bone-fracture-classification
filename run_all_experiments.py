"""
Bone Fracture Classification - Master Experiment Runner
=======================================================
Runs the full pipeline with one command:
  1. Data loading & feature extraction
  2. Decision Tree training
  3. SVM training
  4. Naive Bayes training
  5. KNN training
  6. XGBoost training
  7. CNN (EfficientNet-B0) training
  8. Comparison table and charts
"""

import os
import time
import numpy as np
import joblib

from config import RESULTS_DIR, MODELS_DIR
from data_loader import get_splits
from feature_extraction import extract_features_batch, fit_scaler_pca, transform_features
from evaluate import (
    evaluate_model,
    print_comparison_table,
    save_results,
    plot_comparison_roc,
    plot_comparison_bar,
)


def run_classical_models():
    """Run classical ML models with shared feature extraction."""

    print("\n" + "=" * 70)
    print("  PHASE 1: DATA LOADING & FEATURE EXTRACTION")
    print("=" * 70)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits()

    print("\n[INFO] Feature extraction (Train)...")
    F_train = extract_features_batch(X_train)
    print("[INFO] Feature extraction (Val)...")
    F_val = extract_features_batch(X_val)
    print("[INFO] Feature extraction (Test)...")
    F_test = extract_features_batch(X_test)

    # StandardScaler + PCA
    print("\n[INFO] StandardScaler + PCA fit...")
    scaler, pca, F_train_pca = fit_scaler_pca(F_train)
    F_val_pca = transform_features(F_val, scaler, pca)
    F_test_pca = transform_features(F_test, scaler, pca)

    # Save reusable features
    feat_path = os.path.join(RESULTS_DIR, "features.npz")
    np.savez_compressed(
        feat_path,
        F_train_pca=F_train_pca,
        y_train=y_train,
        F_val_pca=F_val_pca,
        y_val=y_val,
        F_test_pca=F_test_pca,
        y_test=y_test,
    )
    joblib.dump({"scaler": scaler, "pca": pca}, os.path.join(MODELS_DIR, "scaler_pca.pkl"))
    print(f"[INFO] Features saved: {feat_path}")

    results = {}

    # Decision Tree
    print("\n" + "=" * 70)
    print("  MODEL 1: DECISION TREE")
    print("=" * 70)

    from train_decision_tree import train_decision_tree, plot_feature_importance

    dt_model, dt_params = train_decision_tree(F_train_pca, y_train, F_val_pca, y_val)
    y_pred = dt_model.predict(F_test_pca)
    y_prob = dt_model.predict_proba(F_test_pca)[:, 1]
    result = evaluate_model("Decision Tree", y_test, y_pred, y_prob)
    result["best_params"] = dt_params
    results["Decision Tree"] = result
    plot_feature_importance(dt_model)
    joblib.dump(
        {"model": dt_model, "scaler": scaler, "pca": pca},
        os.path.join(MODELS_DIR, "decision_tree.pkl"),
    )

    # SVM
    print("\n" + "=" * 70)
    print("  MODEL 2: SVM")
    print("=" * 70)

    from train_svm import train_svm, get_svm_scores

    svm_model, svm_params = train_svm(F_train_pca, y_train, F_val_pca, y_val)
    y_pred = svm_model.predict(F_test_pca)
    y_prob = get_svm_scores(svm_model, F_test_pca)
    result = evaluate_model("SVM", y_test, y_pred, y_prob)
    result["best_params"] = svm_params
    results["SVM"] = result
    joblib.dump(
        {"model": svm_model, "scaler": scaler, "pca": pca},
        os.path.join(MODELS_DIR, "svm.pkl"),
    )

    # Naive Bayes
    print("\n" + "=" * 70)
    print("  MODEL 3: NAIVE BAYES")
    print("=" * 70)

    from train_naive_bayes import train_naive_bayes

    nb_model, nb_params = train_naive_bayes(F_train_pca, y_train, F_val_pca, y_val)
    y_pred = nb_model.predict(F_test_pca)
    y_prob = nb_model.predict_proba(F_test_pca)[:, 1]
    result = evaluate_model("Naive Bayes", y_test, y_pred, y_prob)
    result["best_params"] = nb_params
    results["Naive Bayes"] = result
    joblib.dump(
        {"model": nb_model, "scaler": scaler, "pca": pca},
        os.path.join(MODELS_DIR, "naive_bayes.pkl"),
    )

    # KNN
    print("\n" + "=" * 70)
    print("  MODEL 4: KNN")
    print("=" * 70)

    from train_knn import train_knn

    knn_model, knn_params = train_knn(F_train_pca, y_train, F_val_pca, y_val)
    y_pred = knn_model.predict(F_test_pca)
    y_prob = knn_model.predict_proba(F_test_pca)[:, 1]
    result = evaluate_model("KNN", y_test, y_pred, y_prob)
    result["best_params"] = knn_params
    results["KNN"] = result
    joblib.dump(
        {"model": knn_model, "scaler": scaler, "pca": pca},
        os.path.join(MODELS_DIR, "knn.pkl"),
    )

    # XGBoost
    print("\n" + "=" * 70)
    print("  MODEL 5: XGBOOST")
    print("=" * 70)

    from train_xgboost import train_xgboost, plot_xgb_feature_importance

    xgb_model, xgb_params = train_xgboost(F_train_pca, y_train, F_val_pca, y_val)
    y_pred = xgb_model.predict(F_test_pca)
    y_prob = xgb_model.predict_proba(F_test_pca)[:, 1]
    result = evaluate_model("XGBoost", y_test, y_pred, y_prob)
    result["best_params"] = xgb_params
    results["XGBoost"] = result
    plot_xgb_feature_importance(xgb_model)
    joblib.dump(
        {"model": xgb_model, "scaler": scaler, "pca": pca},
        os.path.join(MODELS_DIR, "xgboost.pkl"),
    )

    return results, y_test


def run_cnn():
    """Run CNN model."""
    print("\n" + "=" * 70)
    print("  MODEL 6: CNN (EfficientNet-B0)")
    print("=" * 70)

    from train_cnn import main as cnn_main

    return cnn_main()


def main():
    t0 = time.time()

    print("#" * 70)
    print("#  BONE FRACTURE CLASSIFICATION - FULL EXPERIMENT PIPELINE  #")
    print("#" * 70)

    all_results = {}

    # Classical ML models
    classical_results, _ = run_classical_models()
    all_results.update(classical_results)

    # CNN
    cnn_result = run_cnn()
    all_results["CNN (EfficientNet-B0)"] = cnn_result

    # Comparison
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)

    print_comparison_table(all_results)
    plot_comparison_roc(all_results)
    plot_comparison_bar(all_results)

    # Save JSON summary
    save_results(all_results)

    elapsed = time.time() - t0
    print(f"\n[INFO] Total elapsed time: {elapsed / 60:.1f} minutes")
    print("#" * 70)
    print("#  EXPERIMENTS COMPLETED                                      #")
    print("#" * 70)

    return all_results


if __name__ == "__main__":
    all_results = main()
