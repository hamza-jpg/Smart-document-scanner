"""Domain-specific exceptions for document scanning."""

class DocumentScannerError(Exception):
    """Base exception for all document scanner errors."""
    pass

class ImageLoadError(DocumentScannerError):
    """Raised when an image cannot be read or is invalid."""
    pass

class DocumentNotFoundError(DocumentScannerError):
    """Raised when no valid document contour can be detected."""
    pass

class GeometryError(DocumentScannerError):
    """Raised when geometric operations or homography calculation fails."""
    pass

class FilterError(DocumentScannerError):
    """Raised when post-processing filter application fails."""
    pass
