"""
Bone Fracture Classification - Feature Extraction
=================================================
Feature pipeline for classical ML models:
  - HOG
  - LBP
  - GLCM
  - Feature fusion + PCA
"""

import os
import sys
import time

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    GLCM_ANGLES,
    GLCM_DISTANCES,
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    IMAGE_SIZE,
    LBP_N_POINTS,
    LBP_RADIUS,
    PCA_VARIANCE_RATIO,
)


def extract_hog(img: np.ndarray) -> np.ndarray:
    """Extract HOG features with OpenCV."""
    win_size = IMAGE_SIZE
    block_size = (
        HOG_CELLS_PER_BLOCK[0] * HOG_PIXELS_PER_CELL[0],
        HOG_CELLS_PER_BLOCK[1] * HOG_PIXELS_PER_CELL[1],
    )
    block_stride = HOG_PIXELS_PER_CELL
    cell_size = HOG_PIXELS_PER_CELL
    nbins = HOG_ORIENTATIONS

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    descriptor = hog.compute(img)
    return descriptor.flatten().astype(np.float32)


def extract_lbp(img: np.ndarray) -> np.ndarray:
    """Extract uniform LBP histogram."""
    lbp = local_binary_pattern(img, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_glcm(img: np.ndarray) -> np.ndarray:
    """Extract GLCM texture statistics."""
    img_q = (img / 4).astype(np.uint8)
    angles_rad = [np.deg2rad(angle) for angle in GLCM_ANGLES]

    glcm = graycomatrix(
        img_q,
        distances=GLCM_DISTANCES,
        angles=angles_rad,
        levels=64,
        symmetric=True,
        normed=True,
    )

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]

    features = []
    for prop in properties:
        values = graycoprops(glcm, prop)
        features.extend(values.flatten())

    return np.array(features, dtype=np.float32)


def extract_all_features(img: np.ndarray) -> np.ndarray:
    """Concatenate HOG, LBP, and GLCM features."""
    hog_feat = extract_hog(img)
    lbp_feat = extract_lbp(img)
    glcm_feat = extract_glcm(img)
    return np.concatenate([hog_feat, lbp_feat, glcm_feat])


def extract_features_batch(images: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Extract features for a batch of grayscale images."""
    features = []
    total = len(images)
    start_time = time.time()

    for idx, img in enumerate(images, start=1):
        features.append(extract_all_features(img))

        if verbose and (idx % 50 == 0 or idx == total):
            ratio = idx / max(total, 1)
            width = 30
            filled = int(width * ratio)
            bar = "#" * filled + "-" * (width - filled)
            elapsed = time.time() - start_time
            sys.stdout.write(
                f"\r  Feature extraction: [{bar}] {idx}/{total} ({ratio * 100:5.1f}%) | {elapsed:6.1f}s"
            )
            sys.stdout.flush()

    if verbose:
        print()

    return np.array(features, dtype=np.float32)


def fit_scaler_pca(X_train: np.ndarray, variance_ratio: float = PCA_VARIANCE_RATIO):
    """Fit StandardScaler and PCA on training features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=variance_ratio, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(
        f"[INFO] PCA: {X_train.shape[1]} -> {pca.n_components_} components "
        f"({variance_ratio * 100:.0f}% variance)"
    )

    return scaler, pca, X_pca


def transform_features(X: np.ndarray, scaler, pca) -> np.ndarray:
    """Transform features with fitted scaler and PCA."""
    X_scaled = scaler.transform(X)
    return pca.transform(X_scaled)


if __name__ == "__main__":
    from config import FRACTURED_DIR

    test_files = [
        fname
        for fname in os.listdir(FRACTURED_DIR)
        if fname.lower().endswith(".jpg") and not fname.startswith("AUG_")
    ][:1]

    if test_files:
        path = os.path.join(FRACTURED_DIR, test_files[0])
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMAGE_SIZE)

        hog_f = extract_hog(img)
        lbp_f = extract_lbp(img)
        glcm_f = extract_glcm(img)
        all_f = extract_all_features(img)

        print(f"HOG shape: {hog_f.shape}")
        print(f"LBP shape: {lbp_f.shape}")
        print(f"GLCM shape: {glcm_f.shape}")
        print(f"Total shape: {all_f.shape}")
