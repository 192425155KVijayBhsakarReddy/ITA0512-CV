"""
evaluate.py
===========
Evaluation and visualization module for comparing Traditional Computer Vision
vs Deep Learning on Industrial Defect Detection.

Generates the 4 required report figures:
1. training_curves.png   — Training vs. validation accuracy and loss over epochs.
2. metrics_comparison.png — Grouped bar chart comparing Accuracy, Precision, Recall, and F1.
3. confusion_matrix.png  — Confusion matrix heatmap for the deep learning model (ConfusionMatrixDisplay).
4. sample_detections.png — Grid of 6 test images with predicted label + confidence scores.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Configure clean publication aesthetic
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 1.0


def plot_training_curves(history, save_path="training_curves.png"):
    """
    Plots training vs validation accuracy and loss over epochs (two subplots).
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    # Subplot 1: Loss curves
    ax1.plot(epochs, history['train_loss'], 'o-', color='#e74c3c', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 's--', color='#c0392b', linewidth=2, label='Validation Loss')
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True, facecolor='#f8f9fa')

    # Subplot 2: Accuracy curves
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'o-', color='#2ecc71', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 's--', color='#27ae60', linewidth=2, label='Validation Accuracy')
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True, facecolor='#f8f9fa')

    plt.suptitle("Deep Learning (MobileNetV2) Training & Validation Convergence", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Evaluation] Saved training curves to '{save_path}'")


def plot_metrics_comparison(trad_results, dl_results, save_path="metrics_comparison.png"):
    """
    Generates a grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score
    between Traditional CV and Deep Learning pipelines.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    trad_scores = [
        trad_results['accuracy'] * 100,
        trad_results['precision'] * 100,
        trad_results['recall'] * 100,
        trad_results['f1_score'] * 100
    ]
    dl_scores = [
        dl_results['accuracy'] * 100,
        dl_results['precision'] * 100,
        dl_results['recall'] * 100,
        dl_results['f1_score'] * 100
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    rects1 = ax.bar(x - width/2, trad_scores, width, label='Traditional CV (OpenCV + SVM)',
                    color='#3498db', edgecolor='#2980b9', linewidth=1.2, alpha=0.9)
    rects2 = ax.bar(x + width/2, dl_scores, width, label='Deep Learning (MobileNetV2)',
                    color='#e67e22', edgecolor='#d35400', linewidth=1.2, alpha=0.9)

    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Traditional CV vs. Deep Learning', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='#f8f9fa', fontsize=10, loc='upper left')

    # Value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Inset latency comparison badge
    latency_text = (f"Inference Latency:\n"
                    f"• Traditional CV: {trad_results['mean_latency_ms']:.2f} ms/img\n"
                    f"• Deep Learning:  {dl_results['mean_latency_ms']:.2f} ms/img")
    ax.text(0.98, 0.05, latency_text, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f1f2f6', edgecolor='#ced6e0', alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Evaluation] Saved metrics comparison chart to '{save_path}'")


def plot_confusion_matrix(y_true, y_pred, class_names=('OK', 'Defect'), save_path="confusion_matrix.png"):
    """
    Renders confusion matrix heatmap for the deep learning model on the test set
    using sklearn.metrics.ConfusionMatrixDisplay.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')

    ax.set_title("Deep Learning Confusion Matrix (Test Set)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to '{save_path}'")


def plot_sample_detections(test_paths, y_true, y_pred, y_probs,
                           class_names=('OK', 'Defect'),
                           save_path="sample_detections.png"):
    """
    Renders a 2x3 grid of 6 test images with predicted label + confidence score
    overlaid as titles, mixing correct detections and any misclassified examples.
    """
    num_samples = min(6, len(test_paths))

    # Identify correct vs misclassified indices
    correct_indices = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    misclass_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    selected_indices = []
    # Prioritize showing misclassifications if they exist
    if misclass_indices:
        selected_indices.extend(misclass_indices[:2])
    
    # Fill remaining slots with diverse correct examples (both OK and Defect)
    for idx in correct_indices:
        if len(selected_indices) >= num_samples:
            break
        if idx not in selected_indices:
            selected_indices.append(idx)

    # Fallback if list still short
    for i in range(len(test_paths)):
        if len(selected_indices) >= num_samples:
            break
        if i not in selected_indices:
            selected_indices.append(i)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    for i, idx in enumerate(selected_indices[:6]):
        ax = axes[i // 3, i % 3]
        img_bgr = cv2.imread(test_paths[idx])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        true_cls = class_names[y_true[idx]]
        pred_cls = class_names[y_pred[idx]]
        conf = float(y_probs[idx][y_pred[idx]]) * 100.0

        is_correct = (y_true[idx] == y_pred[idx])
        border_color = '#27ae60' if is_correct else '#e74c3c'
        status_text = "CORRECT" if is_correct else "MISCLASSIFIED"

        ax.imshow(img_rgb)
        title_str = f"True: {true_cls} | Pred: {pred_cls}\nConf: {conf:.1f}% ({status_text})"
        ax.set_title(title_str, fontsize=10, fontweight='bold',
                     color='#1b5e20' if is_correct else '#b71c1c', pad=8)
        ax.axis('off')

        # Add colored border indicating status
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)

    plt.suptitle("Sample Test Detections & Confidence Scores (MobileNetV2)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Evaluation] Saved sample detections grid to '{save_path}'")
