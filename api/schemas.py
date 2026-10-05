"""Pydantic schemas for FastAPI endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field

from docscan.config import FilterMode


class CornerPoint(BaseModel):
    x: float = Field(..., description="X coordinate (column) in original image space")
    y: float = Field(..., description="Y coordinate (row) in original image space")


class DetectionResponse(BaseModel):
    status: str = "success"
    corners: List[List[float]] = Field(
        ...,
        description="4 corner points [TL, TR, BR, BL] in [[x, y], ...] format"
    )
    image_width: int
    image_height: int
    algorithm: str
    message: Optional[str] = None


class ScanProcessRequest(BaseModel):
    corners: Optional[List[List[float]]] = Field(
        None,
        description="Optional custom 4 corner coordinates [[x,y],...]. If omitted, auto-detected."
    )
    filter_mode: FilterMode = Field(
        default=FilterMode.BW,
        description="Filter to apply: original, grayscale, bw, or color_enhanced"
    )
    output_format: str = Field(
        default="jpg",
        description="Output format: jpg or png"
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    service: str = "smart-document-scanner-api"
