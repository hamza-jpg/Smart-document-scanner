"""Document contour and corner detection algorithms."""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np

from docscan.config import ScanConfig
from docscan.exceptions import DocumentNotFoundError
from docscan.geometry import simplify_to_quadrilateral, order_points

logger = logging.getLogger(__name__)


def compute_auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Computes Canny edges using dynamic Otsu/median thresholding."""
    median_val = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    return cv2.Canny(gray, lower, upper)


def assess_image_complexity(
    resized_gray: np.ndarray,
    config: ScanConfig
) -> Tuple[float, float, bool]:
    """
    Evaluates global image contrast and background edge clutter.
    Returns: (contrast_std, edge_density_percent, is_complex)
    """
    contrast_std = float(np.std(resized_gray))

    # Fast canny to measure high-frequency clutter
    edges = cv2.Canny(resized_gray, 50, 150)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = (edge_pixels / total_pixels) * 100.0

    is_complex = (
        contrast_std < config.contrast_low_threshold or
        edge_density > config.clutter_density_threshold
    )
    logger.debug(
        f"Pre-flight Assessment: Contrast={contrast_std:.1f}, "
        f"Edge Density={edge_density:.1f}%, Complex={is_complex}"
    )
    return contrast_std, edge_density, is_complex


def is_valid_quadrilateral(quad: np.ndarray, min_area: float) -> bool:
    """Verifies that 4 points form a sensible, non-degenerate convex polygon."""
    if quad is None or len(quad) != 4:
        return False

    ordered = order_points(quad)
    int_quad = ordered.astype(np.int32).reshape(-1, 1, 2)

    # 1. Must be strictly convex
    if not cv2.isContourConvex(int_quad):
        return False

    # 2. Must meet minimum area criteria
    area = cv2.contourArea(int_quad)
    if area < min_area:
        return False

    return True


def detect_fast_edges(
    resized_bgr: np.ndarray,
    config: ScanConfig
) -> Optional[np.ndarray]:
    """Fast detection using Gaussian blur, auto-Canny, and morphological closing."""
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.blur_kernel_size, 0)
    edged = compute_auto_canny(blurred, sigma=config.auto_canny_sigma)

    # Dilate / close to bridge broken contour lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    total_area = resized_bgr.shape[0] * resized_bgr.shape[1]
    min_area = total_area * config.min_area_ratio

    # Sort largest contours first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break

        try:
            quad = simplify_to_quadrilateral(cnt)
            if is_valid_quadrilateral(quad, min_area):
                return quad
        except Exception:
            continue

    return None


def detect_fallback_segmentation(
    resized_bgr: np.ndarray,
    config: ScanConfig
) -> Optional[np.ndarray]:
    """
    Robust fallback segmentation using morphological luminance thresholding
    followed by GrabCut with guaranteed background margin.
    """
    h, w = resized_bgr.shape[:2]
    total_area = h * w
    min_area = total_area * config.min_area_ratio

    # 1. Otsu thresholding on blurred grayscale channel
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:5]:
            if cv2.contourArea(cnt) < min_area:
                break
            try:
                quad = simplify_to_quadrilateral(cnt)
                if is_valid_quadrilateral(quad, min_area):
                    return quad
            except Exception:
                pass

    # 2. GrabCut with guaranteed 4% border margin for definite background
    try:
        margin_x = max(2, int(w * 0.04))
        margin_y = max(2, int(h * 0.04))
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(resized_bgr, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")

        fg_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(fg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            quad = simplify_to_quadrilateral(largest)
            if is_valid_quadrilateral(quad, min_area):
                return quad
    except Exception as e:
        logger.warning(f"GrabCut fallback encountered error: {e}")

    return None


def detect_document(
    image: np.ndarray,
    config: ScanConfig = ScanConfig()
) -> Tuple[np.ndarray, str]:
    """
    Main detection entrypoint:
    1. Scales image down to max_processing_dim for real-time responsiveness.
    2. Runs pre-flight assessment.
    3. Routes to Fast Canny or Fallback Segmentation.
    4. Scales corners back to original image dimensions.
    Returns: (corners_ndarray_4x2, algorithm_name)
    """
    if image is None or image.size == 0:
        raise DocumentNotFoundError("Provided image is empty or invalid.")

    orig_h, orig_w = image.shape[:2]
    max_dim = max(orig_h, orig_w)

    if max_dim > config.max_processing_dim:
        scale = config.max_processing_dim / float(max_dim)
        proc_w = int(orig_w * scale)
        proc_h = int(orig_h * scale)
        resized = cv2.resize(image, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        resized = image.copy()

    resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, _, is_complex = assess_image_complexity(resized_gray, config)

    quad = None
    strategy_used = "fast_edge"

    if not is_complex:
        logger.info("Routing to Fast Edge Detector")
        quad = detect_fast_edges(resized, config)

    if quad is None:
        logger.info("Fast detector yielded no valid contour. Routing to Fallback Segmentation")
        strategy_used = "fallback_segmentation"
        quad = detect_fallback_segmentation(resized, config)

    if quad is None:
        logger.warning("All automated detectors failed. Providing 5% inset fallback rectangle.")
        strategy_used = "inset_default"
        # 5% inset fallback so user can easily adjust on UI canvas
        inset_w = orig_w * 0.05
        inset_h = orig_h * 0.05
        return np.array([
            [inset_w, inset_h],
            [orig_w - inset_w, inset_h],
            [orig_w - inset_w, orig_h - inset_h],
            [inset_w, orig_h - inset_h]
        ], dtype="float32"), strategy_used

    # Scale corners back to original full resolution
    original_quad = quad / scale
    return order_points(original_quad), strategy_used
