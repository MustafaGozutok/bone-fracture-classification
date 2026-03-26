"""
Bone Fracture Classification — CNN (EfficientNet-B0 Transfer Learning)
=======================================================================
SOTA Yaklaşım:
  • EfficientNet-B0 backbone (ImageNet pretrained)
  • OpenCV CLAHE → 3-kanal dönüşüm → ImageNet normalizasyon
  • 2-aşamalı fine-tuning: freeze → unfreeze
  • CosineAnnealingLR scheduler
  • Label Smoothing loss
  • Grad-CAM interpretiability
"""

import os
import time
import copy
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from config import (
    RANDOM_SEED, RESULTS_DIR, MODELS_DIR, IMAGE_SIZE,
    CNN_BATCH_SIZE, CNN_EPOCHS, CNN_LR, CNN_LR_FINETUNE,
    CNN_WEIGHT_DECAY, CNN_FREEZE_EPOCHS, CLASS_NAMES,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE
)
from data_loader import collect_samples, leak_free_split
from evaluate import evaluate_model

# ─── Reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset — OpenCV CLAHE preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

class FractureDataset(Dataset):
    """PyTorch Dataset: OpenCV CLAHE → 3-kanal → transform."""

    def __init__(self, samples, transform=None):
        """
        samples: list of (path, label)
        """
        self.samples = samples
        self.transform = transform
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_SIZE
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # OpenCV ile grayscale yükle (Unicode-safe)
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: siyah görüntü
            img = np.zeros(IMAGE_SIZE, dtype=np.uint8)

        # Resize
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # CLAHE
        img = self.clahe.apply(img)

        # Grayscale → 3-kanal (EfficientNet RGB bekler)
        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # uint8 → float32 [0, 1]
        img_3ch = img_3ch.astype(np.float32) / 255.0

        # HWC → CHW (PyTorch formatı)
        img_tensor = torch.from_numpy(img_3ch).permute(2, 0, 1)

        # Transform (normalizasyon vb.)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Oluşturma
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(num_classes=2, freeze_backbone=True):
    """EfficientNet-B0 transfer learning modeli."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Backbone freeze
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Classifier head değiştir
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )

    return model.to(DEVICE)


def unfreeze_model(model, unfreeze_from=-3):
    """Son N feature block'u unfreeze et."""
    # Tüm parametreleri aç
    for param in model.features[unfreeze_from:].parameters():
        param.requires_grad = True
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer):
    """Tek epoch eğitim."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion):
    """Validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, phase_name=""):
    """Tam eğitim döngüsü."""
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  [{phase_name}] Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    print(f"  [{phase_name}] En iyi Val Acc: {best_val_acc:.4f}")
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
#  Prediction & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def predict(model, loader):
    """Test seti tahminleri."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ═══════════════════════════════════════════════════════════════════════════════
#  Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """Grad-CAM ile son conv layer görselleştirme."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook kaydet
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Grad-CAM heatmap üret."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def visualize_gradcam(model, dataset, n_samples=8, save_dir=RESULTS_DIR):
    """Grad-CAM örnekleri görselleştir."""
    # Son conv layer
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        # Grad-CAM
        cam = grad_cam.generate(input_tensor)

        # Orijinal görüntü
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Heatmap
        cam_resized = cv2.resize(cam, IMAGE_SIZE)
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # Orijinal
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(CLASS_NAMES[label], fontsize=9)
        axes[0, i].axis('off')

        # Grad-CAM overlay
        axes[1, i].imshow(overlay)
        axes[1, i].set_title('Grad-CAM', fontsize=9)
        axes[1, i].axis('off')

    fig.suptitle('EfficientNet-B0 Grad-CAM Visualization', fontsize=14)
    fig.tight_layout()

    path = os.path.join(save_dir, "gradcam_samples.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Grad-CAM kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Training History Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history_phase1, history_phase2, save_dir=RESULTS_DIR):
    """Eğitim loss ve accuracy grafikleri."""
    # Birleştir
    train_loss = history_phase1["train_loss"] + history_phase2["train_loss"]
    val_loss   = history_phase1["val_loss"]   + history_phase2["val_loss"]
    train_acc  = history_phase1["train_acc"]  + history_phase2["train_acc"]
    val_acc    = history_phase1["val_acc"]    + history_phase2["val_acc"]

    epochs = range(1, len(train_loss) + 1)
    phase1_end = len(history_phase1["train_loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
    ax1.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label='Unfreeze')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_acc, 'b-', label='Train Acc')
    ax2.plot(epochs, val_acc, 'r-', label='Val Acc')
    ax2.axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label='Unfreeze')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle('EfficientNet-B0 Training History', fontsize=14)
    fig.tight_layout()

    path = os.path.join(save_dir, "cnn_training_history.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Training history kaydedildi: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  CNN (EfficientNet-B0) — Bone Fracture Classification")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ImageNet normalizasyon (tensor zaten [0,1] aralığında)
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # 1. Veri toplama & split
    print("\n[BİLGİ] Veri toplama...")
    samples = collect_samples()
    train_s, val_s, test_s = leak_free_split(samples)

    print(f"[BİLGİ] Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

    # 2. Dataset & DataLoader
    train_dataset = FractureDataset(train_s, transform=imagenet_normalize)
    val_dataset   = FractureDataset(val_s,   transform=imagenet_normalize)
    test_dataset  = FractureDataset(test_s,  transform=imagenet_normalize)

    train_loader = DataLoader(train_dataset, batch_size=CNN_BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=CNN_BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=CNN_BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    # 3. Model oluştur (backbone frozen)
    print("\n[BİLGİ] Model oluşturuluyor (EfficientNet-B0, frozen)...")
    model = create_model(num_classes=2, freeze_backbone=True)

    # Label smoothing loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ─── Phase 1: Classifier-only eğitim ─────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Phase 1: Classifier Training ({CNN_FREEZE_EPOCHS} epochs)")
    print(f"{'='*50}")

    optimizer_phase1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY
    )
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase1, T_max=CNN_FREEZE_EPOCHS
    )

    model, history1 = train_model(
        model, train_loader, val_loader, criterion,
        optimizer_phase1, scheduler_phase1,
        num_epochs=CNN_FREEZE_EPOCHS,
        phase_name="Phase1-Frozen"
    )

    # ─── Phase 2: Fine-tuning (unfreeze son 3 block) ──────────────────────
    finetune_epochs = CNN_EPOCHS - CNN_FREEZE_EPOCHS

    print(f"\n{'='*50}")
    print(f"  Phase 2: Fine-tuning ({finetune_epochs} epochs)")
    print(f"{'='*50}")

    model = unfreeze_model(model, unfreeze_from=-3)

    optimizer_phase2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CNN_LR_FINETUNE, weight_decay=CNN_WEIGHT_DECAY
    )
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=finetune_epochs
    )

    model, history2 = train_model(
        model, train_loader, val_loader, criterion,
        optimizer_phase2, scheduler_phase2,
        num_epochs=finetune_epochs,
        phase_name="Phase2-Finetune"
    )

    # 4. Training history plot
    plot_training_history(history1, history2)

    # 5. Test seti değerlendirme
    y_true, y_pred, y_prob = predict(model, test_loader)
    result = evaluate_model("CNN (EfficientNet-B0)", y_true, y_pred, y_prob)
    result["best_params"] = {
        "backbone": "EfficientNet-B0",
        "lr": CNN_LR,
        "lr_finetune": CNN_LR_FINETUNE,
        "batch_size": CNN_BATCH_SIZE,
        "epochs": CNN_EPOCHS,
        "freeze_epochs": CNN_FREEZE_EPOCHS,
        "label_smoothing": 0.1
    }

    # 6. Grad-CAM
    print("\n[BİLGİ] Grad-CAM görselleştirme...")
    # Normalizasyon siz test dataset kullan
    test_dataset_raw = FractureDataset(test_s, transform=imagenet_normalize)
    visualize_gradcam(model, test_dataset_raw, n_samples=8)

    # 7. Model kaydet
    model_path = os.path.join(MODELS_DIR, "cnn_efficientnet_b0.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n[BİLGİ] Model kaydedildi: {model_path}")

    return result


if __name__ == "__main__":
    result = main()
