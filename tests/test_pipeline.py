"""End-to-end integration tests for document scanner pipeline."""

import cv2
import numpy as np
import pytest

from docscan.config import FilterMode, ScanConfig
from docscan.pipeline import DocumentScannerEngine


def create_synthetic_document_image() -> np.ndarray:
    """Generates a synthetic high-contrast image with a white document on dark background."""
    # Dark textured table background
    canvas = np.full((800, 1000, 3), 40, dtype=np.uint8)

    # Slanted white document (parallelogram/trapezoid)
    doc_pts = np.array([
        [200, 150],
        [800, 200],
        [750, 700],
        [150, 650]
    ], dtype=np.int32)

    cv2.fillPoly(canvas, [doc_pts], (240, 240, 240))

    # Add black sample text lines
    cv2.line(canvas, (250, 250), (700, 280), (10, 10, 10), 4)
    cv2.line(canvas, (250, 320), (680, 350), (10, 10, 10), 4)
    cv2.line(canvas, (250, 390), (600, 420), (10, 10, 10), 4)

    return canvas


def test_pipeline_detection_and_scan():
    image = create_synthetic_document_image()
    engine = DocumentScannerEngine(ScanConfig())

    # 1. Detect corners
    corners, algo = engine.detect_corners(image)
    assert corners.shape == (4, 2)
    assert algo in ["fast_edge", "fallback_segmentation"]

    # Check that detected corners roughly enclose the synthetic document
    assert np.min(corners[:, 0]) < 250
    assert np.max(corners[:, 0]) > 700
    assert np.min(corners[:, 1]) < 250
    assert np.max(corners[:, 1]) > 600

    # 2. Run full scan pipeline
    result = engine.scan(image, corners=corners, mode=FilterMode.BW)
    assert result.warped is not None
    assert result.enhanced is not None
    assert len(result.enhanced.shape) == 2  # Grayscale/BW is 2D
    assert result.enhanced.shape[0] > 100
    assert result.enhanced.shape[1] > 100


def test_pipeline_all_filter_modes():
    image = create_synthetic_document_image()
    engine = DocumentScannerEngine()
    corners, _ = engine.detect_corners(image)

    for mode in FilterMode:
        result = engine.scan(image, corners=corners, mode=mode)
        assert result.enhanced.size > 0
