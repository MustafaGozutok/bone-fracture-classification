"""
Bone Fracture Classification — Proje Yapılandırması
=====================================================
Tüm scriptler bu dosyadaki sabitleri kullanır.
"""

import os

# ─── Yollar ───────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data", "FracAtlas")
FRACTURED_DIR  = os.path.join(DATA_DIR, "images", "Fractured")
NON_FRAC_DIR   = os.path.join(DATA_DIR, "images", "Non_fractured")
CSV_PATH       = os.path.join(DATA_DIR, "dataset_augmented.csv")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
MODELS_DIR     = os.path.join(BASE_DIR, "saved_models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Genel Parametreler ──────────────────────────────────────────────────────
RANDOM_SEED    = 42
IMAGE_SIZE     = (224, 224)          # (genişlik, yükseklik)
TEST_RATIO     = 0.20
VAL_RATIO      = 0.10               # train setinden ayrılır
NUM_CLASSES    = 2
CLASS_NAMES    = ["Non-fractured", "Fractured"]

# ─── Feature Extraction ──────────────────────────────────────────────────────
# HOG parametreleri
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP parametreleri
LBP_RADIUS     = 3
LBP_N_POINTS   = 8 * LBP_RADIUS     # 24

# GLCM parametreleri
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES    = [0, 45, 90, 135]   # derece cinsinden

# PCA — varyans oranı
PCA_VARIANCE_RATIO = 0.95

# CLAHE parametreleri
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE  = (8, 8)

# ─── Model Eğitim ────────────────────────────────────────────────────────────
# CNN
CNN_BACKBONE     = "efficientnet_b0"
CNN_BATCH_SIZE   = 32
CNN_EPOCHS       = 30
CNN_LR           = 1e-3
CNN_LR_FINETUNE  = 1e-5
CNN_WEIGHT_DECAY = 1e-4
CNN_FREEZE_EPOCHS = 5               # classifier-only eğitim epoch sayısı

# ─── Evaluation ──────────────────────────────────────────────────────────────
METRICS_LIST = [
    "accuracy", "precision", "recall", "specificity",
    "f1_score", "auc_roc", "cohen_kappa"
]
