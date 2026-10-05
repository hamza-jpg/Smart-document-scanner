"""Smart Document Scanner Core Package."""

from docscan.config import FilterMode, ScanConfig
from docscan.exceptions import (
    DocumentNotFoundError,
    DocumentScannerError,
    GeometryError,
    ImageLoadError,
)
from docscan.pipeline import DocumentScannerEngine, ScanResult

__version__ = "0.1.0"
__all__ = [
    "ScanConfig",
    "FilterMode",
    "DocumentScannerEngine",
    "ScanResult",
    "DocumentScannerError",
    "ImageLoadError",
    "DocumentNotFoundError",
    "GeometryError",
]
