"""
deep_learning_model.py
======================
Part B — Deep Learning Pipeline (PyTorch + torchvision MobileNetV2 Transfer Learning).

Architecture & Pipeline:
1. Backbone: Pretrained MobileNetV2 feature extractor (inverted residual bottleneck blocks).
2. Transfer Learning: Frozen convolutional base layers to retain low/mid-level visual primitives.
3. Custom Classifier Head: Dropout(0.3) -> Linear(1280, 128) -> ReLU -> BatchNorm1d -> Dropout(0.2) -> Linear(128, 2).
4. Data Augmentation: Random Flips, Random Rotations, Color/Brightness Jitter to simulate shopfloor variations.
5. Optimization: Adam optimizer + CrossEntropyLoss + Early Stopping on validation loss + Learning Rate Scheduler.
6. Evaluation: Test metrics, class probabilities, and per-image inference latency benchmarking.
"""

import time
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class IndustrialDefectMobileNet(nn.Module):
    """
    MobileNetV2 Transfer Learning Model for Industrial Binary Defect Detection.
    """
    def __init__(self, num_classes=2, freeze_backbone=True, pretrained=True):
        super(IndustrialDefectMobileNet, self).__init__()
        
        # Load MobileNetV2 backbone (with graceful fallback if offline)
        try:
            if pretrained:
                weights = models.MobileNet_V2_Weights.DEFAULT
                self.backbone = models.mobilenet_v2(weights=weights)
                print("[Deep Learning] Successfully loaded pretrained ImageNet weights for MobileNetV2.")
            else:
                self.backbone = models.mobilenet_v2(weights=None)
        except Exception as e:
            print(f"[Deep Learning] Notice: Pretrained download unavailable ({e}). Initializing architecture.")
            self.backbone = models.mobilenet_v2(weights=None)

        # Freeze feature extraction layers for transfer learning stability
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # In MobileNetV2, the classifier input dimension is 1280
        in_features = self.backbone.classifier[1].in_features

        # Replace top classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"             - EarlyStopping counter: {self.counter}/{self.patience} (Best Val Loss: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0


def train_deep_learning_model(model,
                               dataloaders,
                               epochs=15,
                               learning_rate=1e-3,
                               weight_decay=1e-4,
                               patience=5,
                               device=None):
    """
    Trains the deep learning model with tracking of per-epoch loss and accuracy,
    adaptive learning rate scheduling, and early stopping.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[Deep Learning] Initializing training on device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels, _ in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_train += inputs.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects / total_train

        # 2. Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels, _ in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = val_running_corrects / total_val

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print(f"[Deep Learning] Early stopping triggered at epoch {epoch}.")
            break

    # Restore best model weights
    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights)

    total_training_time = time.time() - start_time
    print(f"[Deep Learning] Training complete in {total_training_time:.2f}s.")

    return model, history


def evaluate_deep_learning_model(model, dataloader, device=None):
    """
    Evaluates the trained deep learning model on the test dataset.
    Measures accuracy, precision, recall, F1-score, and per-sample latency.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    latencies = []
    all_paths = []

    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            # Benchmark single-sample inference time for realistic shopfloor latency measurement
            for i in range(inputs.size(0)):
                single_input = inputs[i:i+1].to(device)
                
                t0 = time.perf_counter()
                out = model(single_input)
                prob = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(prob))
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000.0)  # ms
                all_preds.append(pred)
                all_probs.append(prob)
                all_labels.append(labels[i].item())
                all_paths.append(paths[i])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mean_latency = float(np.mean(latencies))

    results = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'mean_latency_ms': mean_latency,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs,
        'latencies': latencies,
        'image_paths': all_paths
    }
    return results
