"""FastAPI Application for Document Scanner."""

import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from docscan.config import FilterMode, ScanConfig
from docscan.exceptions import DocumentScannerError, ImageLoadError
from docscan.pipeline import DocumentScannerEngine
from api.schemas import DetectionResponse, HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("docscan-api")

app = FastAPI(
    title="Smart Document Scanner API",
    description="Production-grade API for automated document boundary detection, homography perspective rectification, and document enhancement.",
    version="0.1.0"
)

# Enable CORS for local and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = DocumentScannerEngine(ScanConfig())

# Path to static web assets
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the interactive web client."""
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Smart Document Scanner API is active. Web UI not found."}


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health and readiness check."""
    return HealthResponse()


@app.post("/api/detect", response_model=DetectionResponse, tags=["Scanner"])
async def detect_document_boundary(
    file: UploadFile = File(..., description="Document photograph (JPEG, PNG, WebP)")
):
    """
    Analyzes uploaded photograph and returns 4-corner polygon coordinates [TL, TR, BR, BL]
    in original image space without warping.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file must be an image. Received content-type: {file.content_type}"
        )

    try:
        contents = await file.read()
        image = engine.load_from_bytes(contents)
    except ImageLoadError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected image read error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read image")

    orig_h, orig_w = image.shape[:2]
    try:
        corners, algorithm = engine.detect_corners(image)
        return DetectionResponse(
            status="success",
            corners=corners.tolist(),
            image_width=orig_w,
            image_height=orig_h,
            algorithm=algorithm
        )
    except DocumentScannerError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


@app.post("/api/process", tags=["Scanner"])
async def process_document(
    file: UploadFile = File(..., description="Original document photograph"),
    corners: Optional[str] = Form(
        None,
        description='Optional JSON string of 4 corners: "[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]". If omitted, auto-detected.'
    ),
    filter_mode: str = Form("bw", description="Filter mode: original, grayscale, bw, color_enhanced"),
    output_format: str = Form("jpg", description="Output image extension: jpg or png")
):
    """
    Performs perspective rectification (homography warp) and applies enhancement filter.
    Returns processed image binary with appropriate Content-Type header.
    """
    try:
        contents = await file.read()
        image = engine.load_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image file: {e}")

    # Parse custom corners if supplied from interactive UI
    parsed_corners = None
    if corners:
        try:
            pts_list = json.loads(corners)
            parsed_corners = np.array(pts_list, dtype="float32")
            if parsed_corners.shape != (4, 2):
                raise ValueError("Corners must be a 4x2 list of coordinates.")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid corners parameter: {e}. Expected JSON string of 4 [x, y] coordinates."
            )

    # Validate filter mode
    try:
        mode = FilterMode(filter_mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown filter mode: '{filter_mode}'. Allowed: {[m.value for m in FilterMode]}"
        )

    try:
        scan_result = engine.scan(image, corners=parsed_corners, mode=mode)
        ext = ".png" if output_format.lower() == "png" else ".jpg"
        media_type = "image/png" if ext == ".png" else "image/jpeg"
        encoded = engine.encode_image(scan_result.enhanced, extension=ext)

        return Response(
            content=encoded,
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="scan{ext}"',
                "X-Detection-Algorithm": scan_result.algorithm,
                "X-Processed-Width": str(scan_result.enhanced.shape[1] if len(scan_result.enhanced.shape) > 1 else 0),
                "X-Processed-Height": str(scan_result.enhanced.shape[0])
            }
        )
    except DocumentScannerError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected scanning error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Scanning failed")


def start():
    """CLI launcher for FastAPI server."""
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
