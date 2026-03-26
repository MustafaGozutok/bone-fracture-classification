"""
Bone Fracture Classification — Evaluation Modülü
==================================================
Tüm modeller için ortak değerlendirme araçları:
  • Confusion Matrix (heatmap)
  • Classification Report
  • ROC Curve + AUC
  • Cohen's Kappa, Specificity
  • Karşılaştırma tablosu & grafikleri
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')          # GUI olmadan çalıştırma
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    cohen_kappa_score
)

from config import RESULTS_DIR, CLASS_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrik Hesaplama
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Tüm metrikleri hesapla.
    y_prob:  pozitif sınıf olasılıkları (AUC için gerekli)
    Returns: dict{metric_name: value}
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1_score":    f1_score(y_true, y_pred, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }

    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auc_roc"] = None

    metrics["confusion_matrix"] = cm.tolist()
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Confusion Matrix Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir=RESULTS_DIR):
    """Confusion matrix heatmap kaydet."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Tahmin', fontsize=12)
    ax.set_ylabel('Gerçek', fontsize=12)
    ax.set_title(f'{model_name} — Confusion Matrix', fontsize=14)

    path = os.path.join(save_dir, f"cm_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Confusion matrix kaydedildi: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  ROC Curve Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_curve(y_true, y_prob, model_name, save_dir=RESULTS_DIR):
    """Tek model için ROC eğrisi kaydet."""
    if y_prob is None:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} — ROC Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    path = os.path.join(save_dir, f"roc_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → ROC eğrisi kaydedildi: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Karşılaştırma Grafikleri
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison_roc(all_results, save_dir=RESULTS_DIR):
    """Tüm modelleri tek ROC grafiğinde karşılaştır."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    for i, (model_name, res) in enumerate(all_results.items()):
        if res.get("y_prob") is not None and res.get("y_true") is not None:
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
            auc = res["metrics"]["auc_roc"]
            ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                    label=f'{model_name} (AUC={auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison — All Models', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    path = os.path.join(save_dir, "roc_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Karşılaştırma ROC kaydedildi: {path}")
    return path


def plot_comparison_bar(all_results, save_dir=RESULTS_DIR):
    """Metrik karşılaştırma bar chart."""
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "specificity", "auc_roc", "cohen_kappa"]
    model_names = list(all_results.keys())

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4 * len(metric_keys), 5))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B']

    for j, metric in enumerate(metric_keys):
        values = []
        names = []
        for mn in model_names:
            v = all_results[mn]["metrics"].get(metric)
            if v is not None:
                values.append(v)
                names.append(mn)

        if values:
            bar_colors = [colors[i % len(colors)] for i in range(len(names))]
            bars = axes[j].bar(range(len(names)), values, color=bar_colors)
            axes[j].set_title(metric.replace('_', ' ').title(), fontsize=11)
            axes[j].set_xticks(range(len(names)))
            axes[j].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            axes[j].set_ylim(0, 1.05)
            axes[j].grid(axis='y', alpha=0.3)
            # Değerleri barların üstüne yaz
            for bar, val in zip(bars, values):
                axes[j].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Model Performance Comparison', fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(save_dir, "comparison_bar.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Karşılaştırma bar chart kaydedildi: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Sonuç Tablosu
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(all_results):
    """Konsola karşılaştırma tablosu yazdır."""
    metric_keys = ["accuracy", "precision", "recall", "specificity",
                   "f1_score", "auc_roc", "cohen_kappa"]

    header = f"{'Model':<20}" + "".join(f"{m:<14}" for m in metric_keys)
    print("\n" + "=" * len(header))
    print("  MODEL PERFORMANCE COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for model_name, res in all_results.items():
        row = f"{model_name:<20}"
        for m in metric_keys:
            v = res["metrics"].get(m)
            if v is not None:
                row += f"{v:<14.4f}"
            else:
                row += f"{'N/A':<14}"
        print(row)
    print("=" * len(header))


def save_results(all_results, save_dir=RESULTS_DIR):
    """Sonuçları JSON olarak kaydet."""
    # JSON-serializable hale getir
    serializable = {}
    for model_name, res in all_results.items():
        serializable[model_name] = {
            "metrics": {k: v for k, v in res["metrics"].items()},
            "best_params": res.get("best_params"),
        }

    path = os.path.join(save_dir, "all_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"  → Sonuçlar kaydedildi: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Full Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_name, y_true, y_pred, y_prob=None):
    """
    Tek model için tam değerlendirme.
    Returns: result dict
    """
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation")
    print(f"{'='*50}")

    metrics = compute_metrics(y_true, y_pred, y_prob)

    # Classification report
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Plots
    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_roc_curve(y_true, y_prob, model_name)

    # Özet
    for key in ["accuracy", "precision", "recall", "specificity", "f1_score", "auc_roc", "cohen_kappa"]:
        v = metrics[key]
        print(f"  {key:<15}: {v:.4f}" if v is not None else f"  {key:<15}: N/A")

    return {
        "metrics": metrics,
        "y_true": y_true.tolist() if isinstance(y_true, np.ndarray) else y_true,
        "y_pred": y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
        "y_prob": y_prob.tolist() if isinstance(y_prob, np.ndarray) else y_prob,
    }
