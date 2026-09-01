"""
traditional_cv.py
=================
Part A — Traditional Computer Vision Pipeline (OpenCV + scikit-image + scikit-learn).

Pipeline Workflow:
1. Color Space Conversion: RGB/BGR -> Grayscale.
2. Gaussian Denoising: Low-pass filter to attenuate high-frequency sensor noise.
3. Contrast-Limited Adaptive Histogram Equalization (CLAHE): Normalizes non-uniform illumination.
4. Segmentation: Otsu thresholding with morphological cleanup to isolate anomalies/workpiece.
5. Shape Feature Extraction: Area, Perimeter, Circularity, Aspect Ratio, Solidity, Extent, and 7 Log Hu Moments.
6. Texture Feature Extraction:
   - Gray-Level Co-occurrence Matrix (GLCM): Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
   - Local Binary Patterns (LBP): Uniform rotation-invariant micro-texture histogram.
   - First-order Intensity Statistics: Mean, Variance, Skewness, Kurtosis, Entropy.
7. Classification: Standardized Support Vector Machine (RBF kernel).
"""

import time
import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def preprocess_image(image_bgr):
    """
    Applies sequential classical CV preprocessing steps:
    1. Grayscale conversion: Reduces 3-channel color to luminance plane.
    2. Gaussian Blur (5x5 kernel): Denoises high-frequency sensor artifacts while preserving edges.
    3. CLAHE: Enhances local contrast and mitigates uneven lighting across the workpiece.
    """
    # Step 1: Grayscale conversion
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()

    # Step 2: Gaussian Denoising (kernel size 5x5, sigma=1.0)
    # Mitigates high-frequency camera shot noise before gradient/edge computations
    denoised = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)

    # Step 3: Illumination correction with CLAHE (clipLimit=2.5, 8x8 grid tiles)
    # Adaptive histogram equalization prevents over-amplification of noise in homogenous regions
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return gray, denoised, enhanced


def segment_anomalies(enhanced_img):
    """
    Segments candidate defect regions and workpiece boundaries using Otsu automatic thresholding
    and morphological opening/closing operations.
    """
    # Step 4: Otsu's Thresholding (bimodal automatic threshold selection)
    # Calculates optimum threshold separating foreground/defect structures from background
    _, otsu_thresh = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleaning: Closing to bridge broken defect contours, Opening to eliminate isolated single-pixel noise
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel_3x3, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_3x3, iterations=1)

    return cleaned_mask


def extract_shape_features(mask):
    """
    Extracts geometric and invariant shape descriptors from segmented contours:
    - Contour Area, Perimeter, Circularity (Compactness)
    - Bounding Box Aspect Ratio, Extent, Convex Hull Solidity
    - 7 Invariant Hu Moments (Log-transformed)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Default zeroed feature vector if no contour is detected (14 shape features)
        return np.zeros(14, dtype=np.float32)

    # Select largest defect / structural contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))
    perimeter = float(cv2.arcLength(largest_contour, True))

    # Circularity: 4 * pi * Area / (Perimeter^2) -> 1.0 for perfect circle
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # Bounding rectangle features
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / (h + 1e-6)
    rect_area = float(w * h)
    extent = area / (rect_area + 1e-6)

    # Convex hull & Solidity
    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / (hull_area + 1e-6)

    # Invariant Hu Moments (7 rotation, scale, and translation invariant moments)
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Log transformation for Hu Moments to prevent dynamic range skew: -1 * sign(h) * log10(|h|)
    log_hu = []
    for h_val in hu_moments:
        if abs(h_val) > 1e-12:
            log_hu.append(-1.0 * np.sign(h_val) * np.log10(abs(h_val)))
        else:
            log_hu.append(0.0)

    shape_vec = [area, perimeter, circularity, aspect_ratio, extent, solidity] + log_hu
    return np.array(shape_vec, dtype=np.float32)


def extract_texture_features(enhanced_img):
    """
    Computes statistical and spatial texture representations:
    1. Gray-Level Co-occurrence Matrix (GLCM): Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
    2. Local Binary Patterns (LBP): Uniform rotation-invariant texture histogram (P=16, R=2).
    3. First-order intensity statistics: Mean, Standard Deviation, Skewness, Kurtosis, Entropy.
    """
    # 1. GLCM computation (distances 1 & 3 pixels, angles 0, 45, 90, 135 deg)
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    # Quantize to 32 gray levels for compact GLCM representation
    quantized_img = (enhanced_img // 8).astype(np.uint8)
    glcm = graycomatrix(quantized_img, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)

    glcm_contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    glcm_dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
    glcm_homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    glcm_energy = float(np.mean(graycoprops(glcm, 'energy')))
    glcm_correlation = float(np.mean(graycoprops(glcm, 'correlation')))
    glcm_asm = float(np.mean(graycoprops(glcm, 'ASM')))

    glcm_features = [glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm]

    # 2. Local Binary Patterns (LBP) with uniform patterns (P=16 points, R=2 radius)
    lbp = local_binary_pattern(enhanced_img, P=16, R=2, method='uniform')
    n_bins = 16 + 2  # uniform LBP yields P + 2 bins
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    # 3. First-order intensity moments
    flat_pixels = enhanced_img.ravel().astype(np.float64)
    stat_mean = float(np.mean(flat_pixels))
    stat_std = float(np.std(flat_pixels))
    stat_skew = float(skew(flat_pixels))
    stat_kurt = float(kurtosis(flat_pixels))

    # Shannon Entropy
    hist, _ = np.histogram(flat_pixels, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    stat_entropy = float(-np.sum(hist * np.log2(hist)))

    stat_features = [stat_mean, stat_std, stat_skew, stat_kurt, stat_entropy]

    texture_vec = np.concatenate([glcm_features, lbp_hist, stat_features])
    return texture_vec.astype(np.float32)


def extract_full_features(image_bgr):
    """
    Combines preprocessing, segmentation, shape descriptor extraction,
    and texture analysis into a single composite feature vector.
    """
    _, _, enhanced = preprocess_image(image_bgr)
    mask = segment_anomalies(enhanced)
    shape_feats = extract_shape_features(mask)
    texture_feats = extract_texture_features(enhanced)
    return np.concatenate([shape_feats, texture_feats])


class TraditionalCVPipeline:
    """
    End-to-End Traditional Computer Vision Defect Classifier using
    OpenCV feature extractors + Support Vector Machine (SVM).
    """
    def __init__(self, c_val=10.0, kernel='rbf', gamma='scale', random_state=42):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=c_val, kernel=kernel, gamma=gamma, probability=True, random_state=random_state))
        ])
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        shape_names = ["area", "perimeter", "circularity", "aspect_ratio", "extent", "solidity"]
        hu_names = [f"log_hu_{i+1}" for i in range(7)]
        glcm_names = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy", "glcm_correlation", "glcm_asm"]
        lbp_names = [f"lbp_bin_{i}" for i in range(18)]
        stat_names = ["stat_mean", "stat_std", "stat_skew", "stat_kurtosis", "stat_entropy"]
        self.feature_names = shape_names + hu_names + glcm_names + lbp_names + stat_names

    def extract_dataset_features(self, image_paths):
        """Extracts feature vectors for a list of image paths."""
        features = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                raise ValueError(f"Unable to read image at path: {p}")
            feat = extract_full_features(img)
            features.append(feat)
        return np.array(features, dtype=np.float32)

    def fit(self, train_paths, train_labels):
        """Extracts features and trains the SVM model."""
        print(f"[Traditional CV] Extracting features from {len(train_paths)} training images...")
        start_time = time.time()
        X_train = self.extract_dataset_features(train_paths)
        extract_duration = time.time() - start_time
        print(f"[Traditional CV] Feature extraction completed in {extract_duration:.2f}s ({X_train.shape[1]} features/image).")

        print("[Traditional CV] Fitting Support Vector Classifier (RBF Kernel)...")
        self.model.fit(X_train, train_labels)
        print("[Traditional CV] Model training complete.")
        return self

    def predict(self, test_paths):
        """Runs predictions and measures per-image inference latency."""
        latencies = []
        predictions = []
        probabilities = []

        for p in test_paths:
            t0 = time.perf_counter()
            img = cv2.imread(p)
            feat = extract_full_features(img).reshape(1, -1)
            pred = self.model.predict(feat)[0]
            prob = self.model.predict_proba(feat)[0]
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)  # ms
            predictions.append(pred)
            probabilities.append(prob)

        return np.array(predictions), np.array(probabilities), np.array(latencies)

    def evaluate(self, test_paths, test_labels):
        """
        Evaluates test set performance and returns metrics:
        Accuracy, Precision, Recall, F1-Score, Mean Inference Time (ms).
        """
        preds, probs, latencies = self.predict(test_paths)
        
        acc = accuracy_score(test_labels, preds)
        prec = precision_score(test_labels, preds, zero_division=0)
        rec = recall_score(test_labels, preds, zero_division=0)
        f1 = f1_score(test_labels, preds, zero_division=0)
        mean_latency = float(np.mean(latencies))

        results = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'mean_latency_ms': mean_latency,
            'predictions': preds,
            'probabilities': probs,
            'latencies': latencies
        }
        return results
