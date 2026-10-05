"""Post-processing image filters for scanned documents."""

import cv2
import numpy as np

from docscan.config import FilterMode, ScanConfig
from docscan.exceptions import FilterError


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts image to grayscale if it is in color."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adaptive_bw_filter(image: np.ndarray, config: ScanConfig) -> np.ndarray:
    """
    Applies resolution-aware adaptive thresholding.
    Dynamically scales the local neighborhood block size based on image resolution
    to prevent hollow text and salt-and-pepper noise.
    """
    gray = to_grayscale(image)

    # 1. Subtle median blur to suppress paper grain without rounding sharp text edges
    pre_filtered = cv2.medianBlur(gray, 3)

    # 2. Compute dynamic block size based on physical image dimensions
    h, w = gray.shape[:2]
    max_dim = max(h, w)
    block_size = int(round(max_dim * config.adaptive_block_ratio))

    # Block size must be odd and at least min_block_size
    block_size = max(config.min_block_size, block_size)
    if block_size % 2 == 0:
        block_size += 1

    # 3. Gaussian adaptive threshold
    bw = cv2.adaptiveThreshold(
        pre_filtered,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=config.adaptive_c
    )
    return bw


def enhance_color(image: np.ndarray, clip_percent: float = 1.0) -> np.ndarray:
    """
    Magic Color enhancement:
    Transforms image to LAB space and applies CLAHE / histogram stretching
    to the luminance (L) channel. Whitens background paper while preserving
    colored signatures, stamps, and photos.
    """
    if len(image.shape) == 2:
        # Convert grayscale back to BGR for uniform color processing
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    # Merge channels and convert back
    merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def apply_filter(image: np.ndarray, mode: FilterMode, config: ScanConfig) -> np.ndarray:
    """Dispatches filtering according to selected FilterMode."""
    if image is None or image.size == 0:
        raise FilterError("Cannot apply filter to empty image.")

    if mode == FilterMode.ORIGINAL:
        return image.copy()
    elif mode == FilterMode.GRAYSCALE:
        return to_grayscale(image)
    elif mode == FilterMode.BW:
        return adaptive_bw_filter(image, config)
    elif mode == FilterMode.COLOR_ENHANCED:
        return enhance_color(image, clip_percent=config.clip_hist_percent)
    else:
        raise FilterError(f"Unsupported filter mode: {mode}")
