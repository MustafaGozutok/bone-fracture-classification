"""
Bone Fracture Classification — Veri Yükleme & Bölme
=====================================================
• OpenCV ile grayscale yükleme + CLAHE ön-işleme
• Augmented görüntülerin orijinaliyle aynı split'e atanması (data leakage önleme)
• Stratified Train / Validation / Test split
"""

import os
import re
import csv
import cv2
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

from config import (
    FRACTURED_DIR, NON_FRAC_DIR, CSV_PATH,
    IMAGE_SIZE, RANDOM_SEED, TEST_RATIO, VAL_RATIO,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE
)


# ─── CLAHE Ön-İşleme ─────────────────────────────────────────────────────────

def apply_clahe(img_gray: np.ndarray) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) uygula."""
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE
    )
    return clahe.apply(img_gray)


# ─── Görüntü Yükleme ─────────────────────────────────────────────────────────

def imread_unicode(path: str, flags=cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    """
    Non-ASCII (Türkçe karakter vb.) yollar için güvenli imread.
    cv2.imread Unicode yollarla sorun yaşar, bu yüzden
    np.fromfile + cv2.imdecode kullanıyoruz.
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img


def load_image(path: str, size: tuple = IMAGE_SIZE, use_clahe: bool = True) -> np.ndarray:
    """
    OpenCV ile grayscale yükle, resize et, opsiyonel CLAHE uygula.
    Returns: (H, W) uint8 numpy array
    """
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görüntü yüklenemedi: {path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if use_clahe:
        img = apply_clahe(img)
    return img


# ─── Augmented → Orijinal Eşleştirme ─────────────────────────────────────────

AUG_PATTERN = re.compile(r"^AUG_[a-z]+_(.+)$")

def get_original_id(filename: str) -> str:
    """
    'AUG_flip_IMG0000019.jpg' → 'IMG0000019.jpg'
    'IMG0000019.jpg'          → 'IMG0000019.jpg'
    """
    m = AUG_PATTERN.match(filename)
    return m.group(1) if m else filename


# ─── Veri Seti Yükleme ───────────────────────────────────────────────────────

def collect_samples():
    """
    Fractured ve Non-fractured dizinlerinden (dosya_yolu, etiket) listesi topla.
    Returns: list of (path, label)  —  label: 1=Fractured, 0=Non-fractured
    """
    samples = []

    # Fractured
    for fname in os.listdir(FRACTURED_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((os.path.join(FRACTURED_DIR, fname), 1))

    # Non-fractured
    for fname in os.listdir(NON_FRAC_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((os.path.join(NON_FRAC_DIR, fname), 0))

    return samples


def leak_free_split(samples, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """
    Data-leakage önleyen split:
    1. Her görüntünün orijinal ID'sini bul
    2. Orijinal ID'lere göre stratified split yap
    3. Augmented görüntüleri orijinalinin split'ine ata

    Returns: (train_samples, val_samples, test_samples)
             her biri [(path, label), ...] formatında
    """
    # Orijinal ID → tüm (path, label) listesi
    original_groups = defaultdict(list)
    original_labels = {}

    for path, label in samples:
        fname = os.path.basename(path)
        orig_id = get_original_id(fname)
        original_groups[orig_id].append((path, label))
        original_labels[orig_id] = label

    # Benzersiz orijinal ID ve etiketleri
    orig_ids = list(original_labels.keys())
    orig_y = [original_labels[oid] for oid in orig_ids]

    # İlk split: train+val vs test
    ids_trainval, ids_test, _, _ = train_test_split(
        orig_ids, orig_y,
        test_size=test_ratio,
        stratify=orig_y,
        random_state=seed
    )

    # İkinci split: train vs val (val_ratio trainval'den)
    y_trainval = [original_labels[oid] for oid in ids_trainval]
    relative_val = val_ratio / (1 - test_ratio)
    ids_train, ids_val, _, _ = train_test_split(
        ids_trainval, y_trainval,
        test_size=relative_val,
        stratify=y_trainval,
        random_state=seed
    )

    # ID kümeleri
    train_set = set(ids_train)
    val_set   = set(ids_val)
    test_set  = set(ids_test)

    # Tüm örnekleri split'lere dağıt
    train_samples, val_samples, test_samples = [], [], []
    for oid in train_set:
        train_samples.extend(original_groups[oid])
    for oid in val_set:
        val_samples.extend(original_groups[oid])
    for oid in test_set:
        test_samples.extend(original_groups[oid])

    return train_samples, val_samples, test_samples


# ─── Batch Yükleme ───────────────────────────────────────────────────────────

def load_dataset(samples, use_clahe=True):
    """
    Örnek listesinden görüntü ve etiket array'leri yükle.
    Returns: X (N, H, W) uint8, y (N,) int
    """
    images, labels = [], []
    for path, label in samples:
        try:
            img = load_image(path, use_clahe=use_clahe)
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"  [UYARI] {path}: {e}")
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


# ─── Ana Kullanım ────────────────────────────────────────────────────────────

def get_splits(use_clahe=True, verbose=True):
    """
    Tam pipeline: toplama → split → yükleme.
    Returns: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    if verbose:
        print("[BİLGİ] Veri toplama...")
    samples = collect_samples()

    if verbose:
        print(f"[BİLGİ] Toplam örnek: {len(samples)}")

    train_s, val_s, test_s = leak_free_split(samples)

    if verbose:
        print(f"[BİLGİ] Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")
        for name, s in [("Train", train_s), ("Val", val_s), ("Test", test_s)]:
            n_frac = sum(1 for _, l in s if l == 1)
            n_nonfrac = len(s) - n_frac
            print(f"         {name} → Fractured: {n_frac}, Non-fractured: {n_nonfrac}")

    if verbose:
        print("[BİLGİ] Görüntüler yükleniyor (CLAHE={})...".format(use_clahe))

    X_train, y_train = load_dataset(train_s, use_clahe=use_clahe)
    X_val,   y_val   = load_dataset(val_s,   use_clahe=use_clahe)
    X_test,  y_test  = load_dataset(test_s,  use_clahe=use_clahe)

    if verbose:
        print(f"[BİLGİ] Boyutlar → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits()
    print("\n✓ Veri yükleme tamamlandı.")
