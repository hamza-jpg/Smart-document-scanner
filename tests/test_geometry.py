"""Unit tests for geometric functions and point ordering."""

import numpy as np
import pytest
import cv2

from docscan.geometry import (
    order_points,
    calculate_destination_size,
    simplify_to_quadrilateral,
    warp_perspective,
)
from docscan.exceptions import GeometryError


def test_order_points_standard_rectangle():
    # Unordered points: [BR, TL, BL, TR]
    pts = np.array([
        [200, 300],  # Bottom-Right
        [10, 20],    # Top-Left
        [10, 300],   # Bottom-Left
        [200, 20],   # Top-Right
    ], dtype="float32")

    ordered = order_points(pts)

    # Expected: TL, TR, BR, BL
    np.testing.assert_allclose(ordered[0], [10, 20])
    np.testing.assert_allclose(ordered[1], [200, 20])
    np.testing.assert_allclose(ordered[2], [200, 300])
    np.testing.assert_allclose(ordered[3], [10, 300])


def test_order_points_rotated():
    # Rotated trapezoid: ensure 4 distinct corners are preserved without duplication
    pts = np.array([
        [150, 40],   # top
        [260, 160],  # right
        [140, 280],  # bottom
        [40, 150],   # left
    ], dtype="float32")

    ordered = order_points(pts)
    assert len(ordered) == 4
    # All 4 rows must be unique
    assert len(np.unique(ordered, axis=0)) == 4


def test_calculate_destination_size():
    quad = np.array([
        [0, 0],
        [300, 0],
        [300, 400],
        [0, 400]
    ], dtype="float32")

    w, h = calculate_destination_size(quad)
    assert w == 300
    assert h == 400


def test_simplify_to_quadrilateral_from_polygon():
    # Polygon with 8 points (e.g. jagged spiral notebook edge)
    poly = np.array([
        [50, 50], [150, 52], [250, 50],
        [252, 200], [250, 350],
        [150, 348], [50, 350],
        [48, 200]
    ], dtype=np.int32).reshape(-1, 1, 2)

    quad = simplify_to_quadrilateral(poly)
    assert quad.shape == (4, 2)
    # Check that simplified corners capture the bounds
    assert np.min(quad[:, 0]) <= 55
    assert np.max(quad[:, 0]) >= 245
    assert np.min(quad[:, 1]) <= 55
    assert np.max(quad[:, 1]) >= 345


def test_warp_perspective():
    # Create synthetic black canvas with white rectangle in center
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (400, 350), (255, 255, 255), -1)

    corners = np.array([
        [100, 100],
        [400, 100],
        [400, 350],
        [100, 350]
    ], dtype="float32")

    warped = warp_perspective(img, corners)
    assert warped.shape[1] == 300  # width
    assert warped.shape[0] == 250  # height
    # Warped image must be mostly white
    assert np.mean(warped) > 240
