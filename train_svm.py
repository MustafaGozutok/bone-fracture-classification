"""
Bone Fracture Classification - SVM (Support Vector Machine)
===========================================================
Classical pipeline:
  - HOG + LBP + GLCM feature fusion + PCA
  - Optional GPU acceleration with cuML if available
  - Linear and RBF kernel comparison
  - Manual CV search with visible progress output
"""

import os
import time
import joblib
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.svm import SVC as SklearnSVC

from config import MODELS_DIR, RANDOM_SEED
from data_loader import get_splits
from evaluate import evaluate_model
from feature_extraction import extract_features_batch, fit_scaler_pca, transform_features

try:
    import cupy as cp
    from cuml.svm import SVC as CuMLSVC
    GPU_SVM_AVAILABLE = True
except Exception:
    cp = None
    CuMLSVC = None
    GPU_SVM_AVAILABLE = False


def _to_numpy(values):
    """Convert sklearn/cupy/cudf-like outputs to numpy arrays."""
    if isinstance(values, np.ndarray):
        return values
    if cp is not None:
        try:
            if isinstance(values, cp.ndarray):
                return cp.asnumpy(values)
        except Exception:
            pass
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    if hasattr(values, "get"):
        return values.get()
    return np.asarray(values)


def _print_progress(prefix, current, total, start_time, width=32):
    """Simple terminal progress bar without extra dependencies."""
    total = max(total, 1)
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    print(
        f"\r{prefix} [{bar}] {current}/{total} ({ratio * 100:5.1f}%) | {elapsed:6.1f}s",
        end="" if current < total else "\n",
        flush=True,
    )


def _build_cpu_candidates():
    return list(ParameterGrid([
        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", 0.001, 0.01],
            "class_weight": [None, "balanced"],
        },
        {
            "kernel": ["linear"],
            "C": [0.1, 1, 10],
            "class_weight": [None, "balanced"],
        },
    ]))


def _build_gpu_candidates():
    # Keep the GPU search smaller and compatible.
    return list(ParameterGrid([
        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", 0.001, 0.01],
        },
        {
            "kernel": ["linear"],
            "C": [0.1, 1, 10],
        },
    ]))


def _make_cpu_model(params):
    return SklearnSVC(
        random_state=RANDOM_SEED,
        probability=False,
        cache_size=512,
        **params,
    )


def _make_gpu_model(params):
    gpu_params = dict(params)
    if gpu_params.get("gamma") == "scale":
        gpu_params["gamma"] = "scale"
    return CuMLSVC(
        probability=False,
        **gpu_params,
    )


def _fit_score_candidate(params, X_cv, y_cv, cv, backend, progress_state=None):
    scores = []
    for train_idx, val_idx in cv.split(X_cv, y_cv):
        X_tr, X_va = X_cv[train_idx], X_cv[val_idx]
        y_tr, y_va = y_cv[train_idx], y_cv[val_idx]

        if backend == "gpu":
            model = _make_gpu_model(params)
        else:
            model = _make_cpu_model(params)

        model.fit(X_tr, y_tr)
        y_pred = _to_numpy(model.predict(X_va)).ravel()
        scores.append(f1_score(y_va, y_pred, zero_division=0))

        if progress_state is not None:
            progress_state["current"] += 1
            _print_progress(
                "SVM search",
                progress_state["current"],
                progress_state["total"],
                progress_state["start_time"],
            )

    return float(np.mean(scores))


def _fit_best_model(params, X_cv, y_cv, backend):
    if backend == "gpu":
        model = _make_gpu_model(params)
    else:
        model = _make_cpu_model(params)
    model.fit(X_cv, y_cv)
    return model


def train_svm(X_train_pca, y_train, X_val_pca, y_val, prefer_gpu=True):
    """Train SVM with optional GPU backend and visible progress."""

    backend = "gpu" if (prefer_gpu and GPU_SVM_AVAILABLE) else "cpu"
    candidates = _build_gpu_candidates() if backend == "gpu" else _build_cpu_candidates()

    X_cv = np.vstack([X_train_pca, X_val_pca])
    y_cv = np.concatenate([y_train, y_val])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    print(f"[INFO] SVM backend: {'GPU (cuML)' if backend == 'gpu' else 'CPU (scikit-learn)'}")
    if backend == "cpu" and prefer_gpu:
        print("[INFO] GPU SVM backend not found. Falling back to CPU.")
    print(f"[INFO] SVM candidate count: {len(candidates)}")
    print(f"[INFO] CV folds: {cv.get_n_splits()}")

    start_time = time.time()
    best_score = -1.0
    best_params = None
    total_steps = len(candidates) * cv.get_n_splits()
    progress_state = {"current": 0, "total": total_steps, "start_time": start_time}

    for idx, params in enumerate(candidates, start=1):
        score = _fit_score_candidate(params, X_cv, y_cv, cv, backend, progress_state)
        if score > best_score:
            best_score = score
            best_params = params

        summary = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  [INFO] Candidate {idx:02d}/{len(candidates)} | F1={score:.4f} | {summary}")

    print(f"[INFO] Best params: {best_params}")
    print(f"[INFO] Best CV F1: {best_score:.4f}")
    print("[INFO] Refitting best SVM on merged train+val set...")

    fit_start = time.time()
    best_model = _fit_best_model(best_params, X_cv, y_cv, backend)
    print(f"[INFO] Refit finished: {time.time() - fit_start:.1f}s")

    return best_model, best_params


def get_svm_scores(model, X):
    """Return scores for ROC/AUC evaluation."""
    if hasattr(model, "decision_function"):
        try:
            return _to_numpy(model.decision_function(X)).ravel()
        except Exception:
            pass
    if hasattr(model, "predict_proba"):
        try:
            return _to_numpy(model.predict_proba(X))[:, 1]
        except Exception:
            pass
    return _to_numpy(model.predict(X)).astype(np.float32).ravel()


def main():
    print("=" * 60)
    print("  SVM - Bone Fracture Classification")
    print("=" * 60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits()

    print("\n[INFO] Feature extraction (Train)...")
    F_train = extract_features_batch(X_train)
    print("[INFO] Feature extraction (Val)...")
    F_val = extract_features_batch(X_val)
    print("[INFO] Feature extraction (Test)...")
    F_test = extract_features_batch(X_test)

    print("\n[INFO] StandardScaler + PCA fit...")
    scaler, pca, F_train_pca = fit_scaler_pca(F_train)
    F_val_pca = transform_features(F_val, scaler, pca)
    F_test_pca = transform_features(F_test, scaler, pca)

    model, best_params = train_svm(F_train_pca, y_train, F_val_pca, y_val)

    y_pred = _to_numpy(model.predict(F_test_pca)).ravel()
    y_prob = get_svm_scores(model, F_test_pca)

    result = evaluate_model("SVM", y_test, y_pred, y_prob)
    result["best_params"] = best_params

    model_path = os.path.join(MODELS_DIR, "svm.pkl")
    try:
        joblib.dump({"model": model, "scaler": scaler, "pca": pca, "best_params": best_params}, model_path)
        print(f"\n[INFO] Model saved: {model_path}")
    except Exception as exc:
        print(f"\n[WARN] Model could not be serialized with joblib: {exc}")

    return result


if __name__ == "__main__":
    result = main()
