"""
FracAtlas - Fractured Bone X-Ray Data Augmentation Script
==========================================================
Hedef: 717 Fractured görüntüden 4 augmented varyant türeterek
       class imbalance'ı giderme (717 → ~3585 Fractured).

Augmentation Teknikleri:
  1. Horizontal Flip
  2. Random Rotation (±15°)
  3. Brightness / Contrast ayarı
  4. Gaussian Noise ekleme
  5. Random Zoom/Crop

Çıktılar:
  - data/FracAtlas/images/Fractured/AUG_<teknik>_<orijinal_ad>.jpg
  - data/FracAtlas/dataset_augmented.csv
"""

import os
import sys
import csv
import random
import math
import shutil
import time

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
except ImportError:
    print("[HATA] Gerekli kütüphaneler eksik. Kurulum yapılıyor...")
    os.system("pip install pillow numpy")
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np

# ─── Yapılandırma ────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data", "FracAtlas")
FRACTURED_DIR  = os.path.join(DATA_DIR, "images", "Fractured")
CSV_INPUT      = os.path.join(DATA_DIR, "dataset.csv")
CSV_OUTPUT     = os.path.join(DATA_DIR, "dataset_augmented.csv")

SEED           = 42
AUGMENTS_PER_IMAGE = 4   # Her görüntüden kaç varyant türetilecek

random.seed(SEED)
np.random.seed(SEED)

# ─── Augmentation Fonksiyonları ───────────────────────────────────────────────

def aug_horizontal_flip(img: Image.Image) -> Image.Image:
    """Yatay çevirme"""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def aug_rotation(img: Image.Image) -> Image.Image:
    """±15° rastgele döndürme (siyah kenar alanları kırpılır)"""
    angle = random.uniform(-15, 15)
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    return rotated


def aug_brightness_contrast(img: Image.Image) -> Image.Image:
    """Parlaklık ve kontrast jitter"""
    # Parlaklık: 0.80 – 1.20
    brightness_factor = random.uniform(0.80, 1.20)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    # Kontrast: 0.80 – 1.20
    contrast_factor = random.uniform(0.80, 1.20)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img


def aug_gaussian_noise(img: Image.Image) -> Image.Image:
    """Gaussian gürültü ekleme"""
    arr = np.array(img).astype(np.float32)
    sigma = random.uniform(5, 15)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def aug_zoom(img: Image.Image) -> Image.Image:
    """Rastgele zoom-in (crop + resize)"""
    w, h = img.size
    scale = random.uniform(0.85, 0.95)   # %5–15 zoom-in
    new_w = int(w * scale)
    new_h = int(h * scale)
    left   = random.randint(0, w - new_w)
    top    = random.randint(0, h - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BICUBIC)


# Augmentation pipeline listesi: (isim, fonksiyon)
AUGMENTATIONS = [
    ("flip",       aug_horizontal_flip),
    ("rotate",     aug_rotation),
    ("brightness", aug_brightness_contrast),
    ("noise",      aug_gaussian_noise),
    ("zoom",       aug_zoom),
]


# ─── CSV Yardımcı Fonksiyonları ───────────────────────────────────────────────

def load_csv(path: str):
    """CSV'yi oku, satırları ve başlığı döndür"""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    return fieldnames, rows


def save_csv(path: str, fieldnames, rows):
    """CSV'yi yaz"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─── Ana Augmentation Döngüsü ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FracAtlas Data Augmentation")
    print("=" * 60)

    # Mevcut fractured görüntüleri listele
    all_files = os.listdir(FRACTURED_DIR)
    original_images = [
        f for f in all_files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not f.startswith("AUG_")
    ]
    print(f"\n[BİLGİ] Orijinal Fractured görüntüsü: {len(original_images)}")

    # CSV yükle
    fieldnames, csv_rows = load_csv(CSV_INPUT)
    # image_id → row mapping
    csv_map = {row["image_id"]: row for row in csv_rows}

    new_rows = []        # Augmented görüntüler için yeni CSV satırları
    generated_count = 0
    skipped_count = 0

    t0 = time.time()

    for idx, img_name in enumerate(original_images, 1):
        img_path = os.path.join(FRACTURED_DIR, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [UYARI] {img_name} açılamadı: {e}")
            skipped_count += 1
            continue

        # Orijinal CSV satırını bul (referans meta-data için)
        base_row = csv_map.get(img_name, {})

        # AUGMENTS_PER_IMAGE adet varyant seç
        chosen_augs = random.sample(AUGMENTATIONS, min(AUGMENTS_PER_IMAGE, len(AUGMENTATIONS)))

        for aug_name, aug_fn in chosen_augs:
            aug_img_name = f"AUG_{aug_name}_{img_name}"
            aug_img_path = os.path.join(FRACTURED_DIR, aug_img_name)

            # Zaten varsa atla
            if os.path.exists(aug_img_path):
                skipped_count += 1
                continue

            # Augment et ve kaydet
            try:
                aug_img = aug_fn(img)
                aug_img.save(aug_img_path, quality=95)
                generated_count += 1
            except Exception as e:
                print(f"  [UYARI] {aug_name} - {img_name}: {e}")
                continue

            # CSV satırı oluştur (orijinal meta-data'yı kopyala, sadece image_id değişir)
            new_row = dict(base_row) if base_row else {k: "0" for k in fieldnames}
            new_row["image_id"] = aug_img_name
            new_rows.append(new_row)

        # İlerleme göster
        if idx % 100 == 0 or idx == len(original_images):
            elapsed = time.time() - t0
            print(f"  [{idx:4d}/{len(original_images)}] İşlendi | "
                  f"Üretilen: {generated_count} | Süre: {elapsed:.1f}s")

    # ─── Özet ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    total_fractured = len(original_images) + generated_count

    print("\n" + "=" * 60)
    print("  AUGMENTATION TAMAMLANDI")
    print("=" * 60)
    print(f"  Orijinal Fractured    : {len(original_images)}")
    print(f"  Üretilen Augmented    : {generated_count}")
    print(f"  Atlanan (zaten var)   : {skipped_count}")
    print(f"  TOPLAM Fractured      : {total_fractured}")
    print(f"  Non-fractured (değişmedi): 3366")
    print(f"  Yeni oran (F:NF)      : 1:{3366/total_fractured:.2f}")
    print(f"  Toplam süre           : {elapsed:.1f}s")

    # ─── CSV Güncelle ──────────────────────────────────────────────────────────
    print(f"\n[BİLGİ] CSV güncelleniyor → {CSV_OUTPUT}")
    all_rows = csv_rows + new_rows
    save_csv(CSV_OUTPUT, fieldnames, all_rows)

    # Doğrulama
    fractured_in_csv = sum(1 for r in all_rows if r.get("fractured") == "1")
    non_frac_in_csv  = sum(1 for r in all_rows if r.get("fractured") == "0")
    print(f"  CSV'deki Fractured    : {fractured_in_csv}")
    print(f"  CSV'deki Non-fractured: {non_frac_in_csv}")
    print(f"  Toplam satır          : {len(all_rows)}")
    print(f"\n  Çıktı kaydedildi: {CSV_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
