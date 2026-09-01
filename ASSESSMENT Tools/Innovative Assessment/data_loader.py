"""
data_loader.py
==============
Module for loading, splitting, and procedurally generating industrial defect datasets.
Supports local industrial defect datasets (e.g. MVTec AD, Kaggle Casting Product,
NEU Surface Defect) and provides an automated synthetic generator fallback.
"""

import os
import random
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class IndustrialDataset(Dataset):
    """PyTorch Dataset wrapper for industrial defect images."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), img_path


def generate_synthetic_dataset(output_dir="data/synthetic_dataset",
                               num_samples_per_class=120,
                               img_size=(256, 256),
                               seed=42):
    """
    Procedurally synthesizes industrial product images (metal disks/plates)
    with realistic defect patterns (cracks, scratches, blobs, pitting, pinholes)
    and pristine 'ok' samples. Ensures the pipeline can run 100% offline.
    """
    np.random.seed(seed)
    random.seed(seed)

    ok_dir = os.path.join(output_dir, "ok")
    defect_dir = os.path.join(output_dir, "defect")
    os.makedirs(ok_dir, exist_ok=True)
    os.makedirs(defect_dir, exist_ok=True)

    print(f"[DataLoader] Generating synthetic dataset in '{output_dir}'...")
    print(f"             - Samples per class: {num_samples_per_class}")
    print(f"             - Resolution: {img_size[0]}x{img_size[1]}")

    h, w = img_size

    def create_base_workpiece():
        """Creates a realistic metallic base with gradient and brushed texture."""
        base = np.zeros((h, w, 3), dtype=np.uint8)
        # Background dark conveyor / staging table
        bg_intensity = np.random.randint(25, 45)
        base[:] = bg_intensity

        # Draw metallic circular product part
        center = (w // 2 + np.random.randint(-5, 6), h // 2 + np.random.randint(-5, 6))
        radius = int(min(h, w) * 0.40) + np.random.randint(-4, 5)

        # Metallic base color with subtle radial gradient
        metal_base = np.random.randint(160, 200)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - center[0])**2 + (y_grid - center[1])**2)
        mask = dist_from_center <= radius

        # Base metallic intensity with radial shading
        gradient = (1.0 - 0.25 * (dist_from_center / radius))
        gradient = np.clip(gradient, 0.7, 1.1)
        part_surface = (metal_base * gradient).astype(np.uint8)

        # Brushed texture (horizontal / radial fine striations)
        brushed_noise = np.random.normal(0, 7, (h, w))
        part_surface = np.clip(part_surface + brushed_noise, 0, 255).astype(np.uint8)

        # Inner ring / bore hole (common in machined cast parts)
        inner_radius = int(radius * 0.32)
        inner_mask = dist_from_center <= inner_radius
        part_surface[inner_mask] = np.random.randint(55, 85)

        for c in range(3):
            ch = base[:, :, c]
            ch[mask] = part_surface[mask]
            # Outer bevel highlight
            bevel_mask = (dist_from_center <= radius) & (dist_from_center >= radius - 3)
            ch[bevel_mask] = np.clip(ch[bevel_mask] + 35, 0, 255)
            base[:, :, c] = ch

        # Add Gaussian sensor noise
        sensor_noise = np.random.normal(0, 3, base.shape).astype(np.int16)
        base = np.clip(base.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)
        return base, center, radius, inner_radius

    # 1. Generate OK images
    for i in range(num_samples_per_class):
        img, _, _, _ = create_base_workpiece()
        save_path = os.path.join(ok_dir, f"ok_{i:04d}.png")
        cv2.imwrite(save_path, img)

    # 2. Generate Defect images
    defect_types = ['scratch', 'crack', 'pitting', 'blob', 'pinhole']
    for i in range(num_samples_per_class):
        img, center, radius, inner_radius = create_base_workpiece()
        dtype = defect_types[i % len(defect_types)]

        # Defect location placed on the outer valid workpiece band
        angle = np.random.uniform(0, 2 * np.pi)
        r_dist = np.random.uniform(inner_radius * 1.3, radius * 0.85)
        dx = int(center[0] + r_dist * np.cos(angle))
        dy = int(center[1] + r_dist * np.sin(angle))

        if dtype == 'scratch':
            # Linear or curved scratch with dark groove and specular highlight edge
            scratch_len = np.random.randint(25, 60)
            angle_deg = np.random.uniform(0, 360)
            rad = np.radians(angle_deg)
            x2 = int(dx + scratch_len * np.cos(rad))
            y2 = int(dy + scratch_len * np.sin(rad))
            # Dark core
            cv2.line(img, (dx, dy), (x2, y2), (40, 40, 40), thickness=np.random.randint(1, 3), lineType=cv2.LINE_AA)
            # Specular shadow line
            cv2.line(img, (dx + 1, dy + 1), (x2 + 1, y2 + 1), (240, 240, 240), thickness=1, lineType=cv2.LINE_AA)

        elif dtype == 'crack':
            # Jagged random walk crack
            curr_x, curr_y = dx, dy
            steps = np.random.randint(15, 30)
            for _ in range(steps):
                next_x = int(curr_x + np.random.randint(-4, 5))
                next_y = int(curr_y + np.random.randint(-4, 5))
                cv2.line(img, (curr_x, curr_y), (next_x, next_y), (30, 30, 30), thickness=2, lineType=cv2.LINE_AA)
                if np.random.rand() > 0.6:  # crack branching
                    branch_x = int(curr_x + np.random.randint(-6, 7))
                    branch_y = int(curr_y + np.random.randint(-6, 7))
                    cv2.line(img, (curr_x, curr_y), (branch_x, branch_y), (35, 35, 35), thickness=1, lineType=cv2.LINE_AA)
                curr_x, curr_y = next_x, next_y

        elif dtype == 'pitting':
            # Cluster of small oxidation pits
            num_pits = np.random.randint(6, 16)
            for _ in range(num_pits):
                px = int(dx + np.random.normal(0, 10))
                py = int(dy + np.random.normal(0, 10))
                p_rad = np.random.randint(2, 5)
                cv2.circle(img, (px, py), p_rad, (45, 45, 45), -1)
                cv2.circle(img, (px - 1, py - 1), max(1, p_rad - 1), (220, 220, 220), 1)

        elif dtype == 'blob':
            # Oil stain or surface inclusion / discoloration blob
            blob_mask = np.zeros((h, w), dtype=np.uint8)
            blob_rad_x = np.random.randint(8, 20)
            blob_rad_y = np.random.randint(6, 16)
            cv2.ellipse(blob_mask, (dx, dy), (blob_rad_x, blob_rad_y), np.random.randint(0, 180), 0, 360, 255, -1)
            blob_mask = cv2.GaussianBlur(blob_mask, (11, 11), 0)
            intensity_shift = np.random.choice([-60, 70])  # dark oil or bright slag
            for c in range(3):
                ch = img[:, :, c].astype(np.float32)
                ch += (blob_mask / 255.0) * intensity_shift
                img[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        elif dtype == 'pinhole':
            # Deep sharp pinhole casting pore
            cv2.circle(img, (dx, dy), np.random.randint(3, 7), (15, 15, 15), -1)
            cv2.circle(img, (dx + 1, dy + 1), np.random.randint(4, 8), (230, 230, 230), 1)

        save_path = os.path.join(defect_dir, f"defect_{dtype}_{i:04d}.png")
        cv2.imwrite(save_path, img)

    print(f"[DataLoader] Synthetic generation complete ({num_samples_per_class} OK, {num_samples_per_class} Defect).")
    return output_dir


def load_dataset_splits(dataset_dir="data/synthetic_dataset",
                        train_ratio=0.70,
                        val_ratio=0.15,
                        test_ratio=0.15,
                        seed=42):
    """
    Scans a dataset directory for 'ok' and 'defect' classes, partitions them into
    stratified Train (70%), Validation (15%), and Test (15%) splits.
    If directory is missing or empty, synthesizes a dataset automatically.
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # Fallback to synthetic if path does not exist
    if not os.path.exists(dataset_dir):
        print(f"[DataLoader] Dataset path '{dataset_dir}' not found. Generating synthetic dataset...")
        dataset_dir = generate_synthetic_dataset(dataset_dir, seed=seed)

    # Search for ok vs defect folders
    classes = ["ok", "defect"]
    label_map = {"ok": 0, "defect": 1}
    all_paths = []
    all_labels = []

    # Check for subfolder structures
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    # Match standard folder names
    ok_synonyms = {"ok", "good", "normal", "pass", "non_defect", "without_defects"}
    defect_synonyms = {"defect", "defective", "bad", "anomaly", "fail", "crack", "scratch", "with_defects"}

    ok_folder = None
    defect_folder = None

    for d in subdirs:
        dl = d.lower()
        if dl in ok_synonyms or any(s in dl for s in ok_synonyms):
            ok_folder = d
        elif dl in defect_synonyms or any(s in dl for s in defect_synonyms):
            defect_folder = d

    if ok_folder and defect_folder:
        for fname in os.listdir(os.path.join(dataset_dir, ok_folder)):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                all_paths.append(os.path.join(dataset_dir, ok_folder, fname))
                all_labels.append(0)
        for fname in os.listdir(os.path.join(dataset_dir, defect_folder)):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                all_paths.append(os.path.join(dataset_dir, defect_folder, fname))
                all_labels.append(1)
    else:
        # Recursive scan fallback
        for root, _, files in os.walk(dataset_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    full_path = os.path.join(root, f)
                    lower_path = full_path.lower()
                    if any(s in lower_path for s in ok_synonyms):
                        all_paths.append(full_path)
                        all_labels.append(0)
                    elif any(s in lower_path for s in defect_synonyms):
                        all_paths.append(full_path)
                        all_labels.append(1)

    if len(all_paths) < 20 or len(set(all_labels)) < 2:
        print(f"[DataLoader] Insufficient or unbalanced images found ({len(all_paths)} images). Synthesizing dataset...")
        dataset_dir = generate_synthetic_dataset(dataset_dir, seed=seed)
        return load_dataset_splits(dataset_dir, train_ratio, val_ratio, test_ratio, seed)

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    # Stratified split: 70% Train, 30% Temp (15% Val + 15% Test)
    temp_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=temp_ratio,
        random_state=seed,
        stratify=all_labels
    )

    # Split temp into Val (50% of 30% = 15%) and Test (50% of 30% = 15%)
    val_test_ratio = test_ratio / temp_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=temp_labels
    )

    splits = {
        'train': {
            'paths': train_paths.tolist(),
            'labels': train_labels.tolist()
        },
        'val': {
            'paths': val_paths.tolist(),
            'labels': val_labels.tolist()
        },
        'test': {
            'paths': test_paths.tolist(),
            'labels': test_labels.tolist()
        },
        'classes': classes,
        'label_map': label_map,
        'class_counts': {
            'train': {'ok': int(np.sum(train_labels == 0)), 'defect': int(np.sum(train_labels == 1))},
            'val': {'ok': int(np.sum(val_labels == 0)), 'defect': int(np.sum(val_labels == 1))},
            'test': {'ok': int(np.sum(test_labels == 0)), 'defect': int(np.sum(test_labels == 1))}
        }
    }

    print(f"\n[DataLoader] Dataset split summary:")
    print(f"  - Train split: {len(train_paths)} images (OK: {splits['class_counts']['train']['ok']}, Defect: {splits['class_counts']['train']['defect']})")
    print(f"  - Val split:   {len(val_paths)} images (OK: {splits['class_counts']['val']['ok']}, Defect: {splits['class_counts']['val']['defect']})")
    print(f"  - Test split:  {len(test_paths)} images (OK: {splits['class_counts']['test']['ok']}, Defect: {splits['class_counts']['test']['defect']})")

    return splits


def get_pytorch_dataloaders(splits, batch_size=16, img_size=(224, 224), num_workers=0):
    """
    Constructs PyTorch DataLoaders with transfer-learning augmentations for training
    and deterministic transforms for validation/testing.
    """
    # ImageNet standard normalization
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Data augmentation for training: rotation, flips, color/lighting jitter, scaling
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    train_dataset = IndustrialDataset(splits['train']['paths'], splits['train']['labels'], transform=train_transform)
    val_dataset = IndustrialDataset(splits['val']['paths'], splits['val']['labels'], transform=eval_transform)
    test_dataset = IndustrialDataset(splits['test']['paths'], splits['test']['labels'], transform=eval_transform)

    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }
    return loaders
