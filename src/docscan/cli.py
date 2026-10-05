"""Command Line Interface for Document Scanner."""

import argparse
import logging
import sys
from pathlib import Path
import cv2

from docscan.config import FilterMode, ScanConfig
from docscan.pipeline import DocumentScannerEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Document Scanner CLI: Detect, flatten, and enhance documents."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to the input document photograph."
    )
    parser.add_argument(
        "-o", "--output",
        default="scanned_output.jpg",
        type=str,
        help="Path to save the processed document scan (default: scanned_output.jpg)."
    )
    parser.add_argument(
        "-m", "--mode",
        choices=[m.value for m in FilterMode],
        default=FilterMode.BW.value,
        help="Filter mode: bw, grayscale, color_enhanced, or original (default: bw)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display intermediate steps in OpenCV windows."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    engine = DocumentScannerEngine(ScanConfig())

    logger.info(f"Loading image from {input_path}...")
    try:
        image = engine.load_from_path(str(input_path))
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        sys.exit(1)

    logger.info("Detecting document boundaries...")
    corners, algorithm = engine.detect_corners(image)
    logger.info(f"Document detected using strategy: {algorithm}")

    filter_mode = FilterMode(args.mode)
    result = engine.scan(image, corners=corners, mode=filter_mode)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result.enhanced)
    logger.info(f"Successfully saved scanned document to {output_path}")

    if args.debug:
        annotated = image.copy()
        cv2.polylines(annotated, [corners.astype(int)], True, (0, 255, 0), 3)
        cv2.imshow("Detected Corners", annotated)
        cv2.imshow("Warped Scan", result.warped)
        cv2.imshow(f"Enhanced Scan ({args.mode})", result.enhanced)
        logger.info("Press any key on image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
