"""Mathematical and geometric utilities for document transformation."""

import math
from typing import Tuple
import cv2
import numpy as np

from docscan.exceptions import GeometryError


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 quadrilateral points in strict clockwise order:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Uses centroid angle sorting to prevent duplicate assignments and
    failures on rotated or perspective-distorted documents.
    """
    pts = np.asarray(pts, dtype="float32")
    if pts.shape != (4, 2):
        pts = pts.reshape(-1, 2)
        if pts.shape != (4, 2):
            raise GeometryError(f"Expected 4 points, got shape {pts.shape}")

    # Calculate centroid
    center = np.mean(pts, axis=0)

    # Sort points clockwise using screen coordinate system (y grows downward)
    # angle = atan2(y - cy, x - cx)
    # In screen coordinates, angles from -pi to +pi go:
    # Top (-pi/2) -> Right (0) -> Bottom (+pi/2) -> Left (+/-pi)
    def clockwise_angle(pt: np.ndarray) -> float:
        return math.atan2(pt[1] - center[1], pt[0] - center[0])

    sorted_pts = sorted(pts, key=clockwise_angle)
    sorted_pts = np.array(sorted_pts, dtype="float32")

    # Determine which point is closest to origin (0, 0) as Top-Left
    distances_to_origin = np.sum(sorted_pts ** 2, axis=1)
    tl_idx = int(np.argmin(distances_to_origin))

    # Reorder starting from top-left, going clockwise
    ordered = np.zeros((4, 2), dtype="float32")
    for i in range(4):
        ordered[i] = sorted_pts[(tl_idx + i) % 4]

    return ordered


def simplify_to_quadrilateral(contour: np.ndarray) -> np.ndarray:
    """
    Simplifies any arbitrary contour into a clean 4-corner quadrilateral.

    1. Computes Convex Hull to eliminate concave artifacts (e.g., spiral rings).
    2. Uses iterative Douglas-Peucker (approxPolyDP) with adaptive epsilon.
    3. Falls back to Minimum Area Bounding Box (minAreaRect) if polygon is non-quadrilateral.
    """
    if contour is None or len(contour) < 3:
        raise GeometryError("Contour contains fewer than 3 vertices.")

    # 1. Convex hull
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, closed=True)

    if perimeter <= 0:
        raise GeometryError("Contour perimeter is zero or degenerate.")

    # 2. Try varying epsilon ratios from 1.5% to 5% to find exactly 4 points
    for eps_ratio in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, eps_ratio * perimeter, closed=True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    # 3. Robust fallback: Minimum Area Rotated Rectangle
    min_rect = cv2.minAreaRect(hull)
    box_pts = cv2.boxPoints(min_rect)
    return np.asarray(box_pts, dtype="float32")


def calculate_destination_size(quad: np.ndarray) -> Tuple[int, int]:
    """Calculates width and height for top-down perspective transform."""
    tl, tr, br, bl = quad

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(round(max(width_top, width_bottom)))

    height_right = np.linalg.norm(br - tr)
    height_left = np.linalg.norm(bl - tl)
    max_height = int(round(max(height_right, height_left)))

    if max_width < 10 or max_height < 10:
        raise GeometryError(
            f"Calculated dimensions too small: width={max_width}, height={max_height}"
        )

    return max_width, max_height


def warp_perspective(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Warps perspective of the document image given 4 corners.
    Corners can be in any order; they will be strictly ordered automatically.
    """
    if image is None or image.size == 0:
        raise GeometryError("Input image is empty or invalid.")

    ordered_quad = order_points(quad)
    max_width, max_height = calculate_destination_size(ordered_quad)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_quad, dst)
    if matrix is None:
        raise GeometryError("Failed to calculate perspective transform matrix.")

    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped
