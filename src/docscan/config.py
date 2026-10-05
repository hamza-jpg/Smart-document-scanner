"""Configuration dataclasses and enums for docscan."""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class FilterMode(str, Enum):
    ORIGINAL = "original"
    GRAYSCALE = "grayscale"
    BW = "bw"
    COLOR_ENHANCED = "color_enhanced"


@dataclass(frozen=True)
class ScanConfig:
    """Configurable hyperparameters for document scanning pipeline."""
    # Pre-processing & Resizing
    max_processing_dim: int = 800
    blur_kernel_size: Tuple[int, int] = (5, 5)
    auto_canny_sigma: float = 0.33

    # Quality & Heuristic Routing
    contrast_low_threshold: float = 30.0
    clutter_density_threshold: float = 16.0

    # Contour & Polygon Approximation
    min_area_ratio: float = 0.04
    approx_epsilon_ratio: float = 0.02

    # Resolution-aware Adaptive B&W Filter
    adaptive_block_ratio: float = 0.022  # ~2.2% of max image dimension
    adaptive_c: int = 9
    min_block_size: int = 15

    # Color enhancement
    clip_hist_percent: float = 1.0
