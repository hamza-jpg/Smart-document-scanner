"""High-level, stateless orchestration pipeline for document scanning."""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from docscan.config import FilterMode, ScanConfig
from docscan.detector import detect_document
from docscan.exceptions import ImageLoadError
from docscan.filters import apply_filter
from docscan.geometry import warp_perspective, order_points


@dataclass
class ScanResult:
    """Contains results of a complete scan operation."""
    warped: np.ndarray
    enhanced: np.ndarray
    corners: np.ndarray
    algorithm: str
    original_shape: Tuple[int, int, int]


class DocumentScannerEngine:
    """
    Stateless, production-ready document scanning engine.
    Separates detection, homography transformation, and filtering.
    """

    def __init__(self, config: Optional[ScanConfig] = None):
        self.config = config or ScanConfig()

    @staticmethod
    def load_from_bytes(image_bytes: bytes) -> np.ndarray:
        """Decodes raw image bytes into a BGR NumPy array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageLoadError("Failed to decode image from provided byte stream.")
        return img

    @staticmethod
    def load_from_path(file_path: str) -> np.ndarray:
        """Reads an image from local filesystem path safely."""
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageLoadError(f"Could not load image at path: {file_path}")
        return img

    @staticmethod
    def encode_image(image: np.ndarray, extension: str = ".jpg", quality: int = 95) -> bytes:
        """Encodes NumPy array back to image bytes (JPEG, PNG)."""
        params = []
        if extension.lower() in [".jpg", ".jpeg"]:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif extension.lower() == ".png":
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]

        success, encoded = cv2.imencode(extension, image, params)
        if not success:
            raise ImageLoadError(f"Failed to encode image to {extension}")
        return encoded.tobytes()

    def detect_corners(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """Detects 4 document corner coordinates in the given image."""
        return detect_document(image, self.config)

    def warp_document(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Applies perspective transformation to flatten document using 4 corners."""
        ordered = order_points(corners)
        return warp_perspective(image, ordered)

    def apply_enhancement(
        self,
        warped: np.ndarray,
        mode: FilterMode = FilterMode.BW
    ) -> np.ndarray:
        """Applies post-scan enhancements (Adaptive BW, Grayscale, Magic Color)."""
        return apply_filter(warped, mode, self.config)

    def scan(
        self,
        image: np.ndarray,
        corners: Optional[np.ndarray] = None,
        mode: FilterMode = FilterMode.BW
    ) -> ScanResult:
        """
        Executes end-to-end scanning pipeline:
        1. If corners not provided, detects them automatically.
        2. Warps perspective to top-down view.
        3. Applies chosen enhancement filter.
        """
        if corners is None:
            detected_corners, algorithm = self.detect_corners(image)
        else:
            detected_corners = order_points(corners)
            algorithm = "user_provided"

        warped = self.warp_document(image, detected_corners)
        enhanced = self.apply_enhancement(warped, mode)

        return ScanResult(
            warped=warped,
            enhanced=enhanced,
            corners=detected_corners,
            algorithm=algorithm,
            original_shape=image.shape
        )
