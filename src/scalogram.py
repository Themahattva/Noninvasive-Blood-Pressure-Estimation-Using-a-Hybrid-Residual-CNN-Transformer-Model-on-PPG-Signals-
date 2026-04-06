import numpy as np
import pywt
import cv2


def generate_scalogram(ppg_window, scales=128, wavelet='morl'):
    scale_arr = np.arange(1, scales + 1)
    coeffs, _ = pywt.cwt(ppg_window, scale_arr, wavelet)
    magnitude = np.abs(coeffs)
    return magnitude


def postprocess_scalogram(magnitude, size=128, gamma=0.6):
    # Normalize to [0, 1]
    mag_min, mag_max = magnitude.min(), magnitude.max()
    eps = 1e-8
    norm = (magnitude - mag_min) / (mag_max - mag_min + eps)

    # Resize to 128x128
    resized = cv2.resize(norm, (size, size), interpolation=cv2.INTER_LINEAR)

    # Histogram equalization (convert to uint8 first)
    img_uint8 = (resized * 255).astype(np.uint8)
    eq = cv2.equalizeHist(img_uint8)

    # Gamma correction
    eq_float = eq.astype(np.float32) / 255.0
    gamma_corrected = np.power(eq_float, gamma)

    return gamma_corrected


def compute_image_quality(img):
    img_uint8 = (img * 255).astype(np.uint8)
    # Contrast via std
    contrast = float(np.std(img_uint8))
    # Edge strength via Laplacian
    edges = cv2.Laplacian(img_uint8, cv2.CV_64F)
    edge_score = float(np.var(edges))
    return contrast, edge_score


def is_quality_scalogram(img, min_contrast=10.0, min_edge=50.0):
    contrast, edge_score = compute_image_quality(img)
    return contrast >= min_contrast and edge_score >= min_edge


def ppg_to_scalogram(ppg_window, scales=128, wavelet='morl', size=128, gamma=0.6,
                      min_contrast=10.0, min_edge=50.0):
    magnitude = generate_scalogram(ppg_window, scales, wavelet)
    img = postprocess_scalogram(magnitude, size, gamma)
    if not is_quality_scalogram(img, min_contrast, min_edge):
        return None
    return img  # shape: (128, 128), float32 in [0,1]
