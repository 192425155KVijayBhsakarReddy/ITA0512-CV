"""
main.py
=======
Orchestrator for ITA05 Computer Vision Industrial Defect Detection Comparison.

Executes and compares:
1. Part A: Traditional CV Pipeline (OpenCV Preprocessing + Otsu + GLCM/LBP/Hu Features + SVM)
2. Part B: Deep Learning Pipeline (PyTorch MobileNetV2 Transfer Learning + Augmentation)

Generates performance metrics and saves all 4 required report figures:
- training_curves.png
- metrics_comparison.png
- confusion_matrix.png
- sample_detections.png
"""

import os
import argparse
import random
import numpy as np
import torch
import pandas as pd

from data_loader import load_dataset_splits, get_pytorch_dataloaders, generate_synthetic_dataset
from traditional_cv import TraditionalCVPipeline
from deep_learning_model import IndustrialDefectMobileNet, train_deep_learning_model, evaluate_deep_learning_model
from evaluate import (
    plot_training_curves,
    plot_metrics_comparison,
    plot_confusion_matrix,
    plot_sample_detections
)


def set_seed(seed=42):
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Industrial Defect Detection: Traditional CV vs. Deep Learning")
    parser.add_argument("--dataset_dir", type=str, default="data/synthetic_dataset",
                        help="Path to dataset directory containing 'ok' and 'defect' subdirectories.")
    parser.add_argument("--generate_synthetic", action="store_true", default=False,
                        help="Force regeneration of synthetic industrial defect dataset.")
    parser.add_argument("--num_samples", type=int, default=120,
                        help="Number of samples per class when generating synthetic dataset.")
    parser.add_argument("--epochs", type=int, default=12,
                        help="Number of training epochs for deep learning model.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for deep learning DataLoader.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate for Adam optimizer.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save generated comparison plots and evaluation figures.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def print_banner(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print_banner("ITA05: COMPUTER VISION — INDUSTRIAL DEFECT DETECTION COMPARISON")
    print(f"• Dataset directory:   {args.dataset_dir}")
    print(f"• Output directory:    {args.output_dir}")
    print(f"• Epochs:              {args.epochs}")
    print(f"• Batch size:          {args.batch_size}")
    print(f"• Learning rate:       {args.lr}")
    print(f"• Random seed:         {args.seed}")

    # -------------------------------------------------------------
    # 1. Dataset Loading / Synthetic Fallback
    # -------------------------------------------------------------
    print_banner("STEP 1: DATASET INGESTION & 70/15/15 STRATIFIED SPLIT")
    if args.generate_synthetic:
        generate_synthetic_dataset(args.dataset_dir, num_samples_per_class=args.num_samples, seed=args.seed)

    splits = load_dataset_splits(
        dataset_dir=args.dataset_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )

    # -------------------------------------------------------------
    # 2. Part A: Traditional Computer Vision (OpenCV + SVM)
    # -------------------------------------------------------------
    print_banner("STEP 2: PART A — TRADITIONAL CV PIPELINE (OpenCV + SVM)")
    print("Pipeline stages: Grayscale -> Gaussian Filter -> CLAHE -> Otsu -> Contours/Hu Moments -> GLCM/LBP -> SVM")
    
    trad_pipeline = TraditionalCVPipeline(c_val=10.0, kernel='rbf', gamma='scale', random_state=args.seed)
    trad_pipeline.fit(splits['train']['paths'], splits['train']['labels'])
    
    print("\n[Traditional CV] Running evaluation on test set (15% split)...")
    trad_results = trad_pipeline.evaluate(splits['test']['paths'], splits['test']['labels'])
    
    print(f"  • Test Accuracy:      {trad_results['accuracy'] * 100:.2f}%")
    print(f"  • Test Precision:     {trad_results['precision'] * 100:.2f}%")
    print(f"  • Test Recall:        {trad_results['recall'] * 100:.2f}%")
    print(f"  • Test F1-Score:      {trad_results['f1_score'] * 100:.2f}%")
    print(f"  • Avg Latency/image:  {trad_results['mean_latency_ms']:.2f} ms")

    # -------------------------------------------------------------
    # 3. Part B: Deep Learning (PyTorch MobileNetV2 Transfer Learning)
    # -------------------------------------------------------------
    print_banner("STEP 3: PART B — DEEP LEARNING PIPELINE (MobileNetV2)")
    print("Backbone: Pretrained MobileNetV2 | Custom Head: Dropout(0.3) -> Dense(128) -> Dense(2)")
    print("Augmentations: Random Flips, Rotations, Color/Brightness Jitter")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders = get_pytorch_dataloaders(splits, batch_size=args.batch_size, img_size=(224, 224))

    dl_model = IndustrialDefectMobileNet(num_classes=2, freeze_backbone=True, pretrained=True)
    trained_model, history = train_deep_learning_model(
        model=dl_model,
        dataloaders=dataloaders,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=5,
        device=device
    )

    print("\n[Deep Learning] Running evaluation on test set (15% split)...")
    dl_results = evaluate_deep_learning_model(trained_model, dataloaders['test'], device=device)

    print(f"  • Test Accuracy:      {dl_results['accuracy'] * 100:.2f}%")
    print(f"  • Test Precision:     {dl_results['precision'] * 100:.2f}%")
    print(f"  • Test Recall:        {dl_results['recall'] * 100:.2f}%")
    print(f"  • Test F1-Score:      {dl_results['f1_score'] * 100:.2f}%")
    print(f"  • Avg Latency/image:  {dl_results['mean_latency_ms']:.2f} ms")

    # -------------------------------------------------------------
    # 4. Evaluation Visualizations & Export Artifacts
    # -------------------------------------------------------------
    print_banner("STEP 4: GENERATING PUBLICATION-QUALITY COMPARISON FIGURES")
    
    path_curves = os.path.join(args.output_dir, "training_curves.png")
    path_metrics = os.path.join(args.output_dir, "metrics_comparison.png")
    path_cm = os.path.join(args.output_dir, "confusion_matrix.png")
    path_samples = os.path.join(args.output_dir, "sample_detections.png")

    # Also save to current directory for easy root access
    plot_training_curves(history, save_path="training_curves.png")
    plot_training_curves(history, save_path=path_curves)

    plot_metrics_comparison(trad_results, dl_results, save_path="metrics_comparison.png")
    plot_metrics_comparison(trad_results, dl_results, save_path=path_metrics)

    plot_confusion_matrix(dl_results['true_labels'], dl_results['predictions'],
                          class_names=['OK', 'Defect'], save_path="confusion_matrix.png")
    plot_confusion_matrix(dl_results['true_labels'], dl_results['predictions'],
                          class_names=['OK', 'Defect'], save_path=path_cm)

    plot_sample_detections(dl_results['image_paths'], dl_results['true_labels'],
                           dl_results['predictions'], dl_results['probabilities'],
                           class_names=['OK', 'Defect'], save_path="sample_detections.png")
    plot_sample_detections(dl_results['image_paths'], dl_results['true_labels'],
                           dl_results['predictions'], dl_results['probabilities'],
                           class_names=['OK', 'Defect'], save_path=path_samples)

    # -------------------------------------------------------------
    # 5. Formatted Summary Report
    # -------------------------------------------------------------
    print_banner("FINAL BENCHMARK COMPARISON SUMMARY TABLE")
    df_summary = pd.DataFrame([
        {
            "Metric / Pipeline": "Accuracy",
            "Traditional CV (OpenCV + SVM)": f"{trad_results['accuracy'] * 100:.2f}%",
            "Deep Learning (MobileNetV2)": f"{dl_results['accuracy'] * 100:.2f}%"
        },
        {
            "Metric / Pipeline": "Precision",
            "Traditional CV (OpenCV + SVM)": f"{trad_results['precision'] * 100:.2f}%",
            "Deep Learning (MobileNetV2)": f"{dl_results['precision'] * 100:.2f}%"
        },
        {
            "Metric / Pipeline": "Recall",
            "Traditional CV (OpenCV + SVM)": f"{trad_results['recall'] * 100:.2f}%",
            "Deep Learning (MobileNetV2)": f"{dl_results['recall'] * 100:.2f}%"
        },
        {
            "Metric / Pipeline": "F1-Score",
            "Traditional CV (OpenCV + SVM)": f"{trad_results['f1_score'] * 100:.2f}%",
            "Deep Learning (MobileNetV2)": f"{dl_results['f1_score'] * 100:.2f}%"
        },
        {
            "Metric / Pipeline": "Avg Latency per Image",
            "Traditional CV (OpenCV + SVM)": f"{trad_results['mean_latency_ms']:.2f} ms",
            "Deep Learning (MobileNetV2)": f"{dl_results['mean_latency_ms']:.2f} ms"
        }
    ])
    print(df_summary.to_string(index=False))

    csv_path = os.path.join(args.output_dir, "benchmark_metrics.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\n[Summary] Metrics table exported to '{csv_path}'")
    print("[Summary] All tasks completed successfully.")


if __name__ == "__main__":
    main()
