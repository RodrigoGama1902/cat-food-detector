"""Cat Food Detector.

Lightweight, local, no-ML computer vision script that estimates how much
food is present in a cat food bowl from a single image snapshot.

Usage:
    python detector.py latest.jpg

Output (stdout, JSON):
    {"food_present": true, "coverage": 0.63}
"""

import argparse
import json
import sys

import cv2
import numpy as np

from config import DEFAULTS

# --- Configuration -----------------------------------------------------------
# Module-level defaults come from config.py's "day" profile; the UI persists
# per-profile overrides (day/night) to config.json.
_DAY = DEFAULTS["profiles"]["day"]

# Region of Interest as (x, y, width, height) in pixels.
ROI = tuple(DEFAULTS["roi"])

# Grayscale threshold: for the brightness method, pixels darker than this are
# treated as food; for the texture method, this is the Canny edge sensitivity.
THRESHOLD = _DAY["threshold"]

# Coverage above which the bowl is considered to contain food.
MINIMUM_COVERAGE = _DAY["minimum_coverage"]

# Raw coverage that represents a full bowl. Raw coverage is remapped so that
# MINIMUM_COVERAGE -> 0.0 and FULL_COVERAGE -> 1.0 (clamped in between).
FULL_COVERAGE = _DAY["full_coverage"]

# Minimum artifact area (in pixels) to keep. Smaller blobs are removed as noise.
# Set to 0 to disable artifact removal.
MIN_ARTIFACT_AREA = DEFAULTS["min_artifact_area"]

# Detection method: "texture" (edge density, lighting-robust) or "brightness".
METHOD = _DAY["method"]

# Dilation iterations used to fill textured (food) regions in the texture method.
DILATE = _DAY["dilate"]
# -----------------------------------------------------------------------------


def load_image(path):
    """Load an image from disk, raising a clear error if it cannot be read."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def crop_roi(image, roi):
    """Crop the configured ROI, clamped to the image bounds."""
    x, y, w, h = roi
    height, width = image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"ROI {roi} is outside the image bounds {width}x{height}")
    return image[y0:y1, x0:x1]


def compute_mask(
    crop,
    threshold=THRESHOLD,
    min_artifact_area=MIN_ARTIFACT_AREA,
    method=METHOD,
    dilate=DILATE,
):
    """Return the binary food mask for the cropped region."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if method == "brightness":
        # Dark pixels (below threshold) are treated as food.
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        # Texture: food (kibble) creates dense edges; an empty bowl is smooth.
        mask = _texture_mask(gray, threshold, dilate)

    if min_artifact_area > 0:
        mask = remove_small_artifacts(mask, min_artifact_area)
    return mask


def _texture_mask(gray, edge_threshold, dilate):
    """Build a mask of textured regions using Canny edges plus morphology."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, edge_threshold, edge_threshold * 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if dilate > 0:
        edges = cv2.dilate(edges, kernel, iterations=dilate)
    # Close gaps so scattered edges merge into solid food blobs.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges


def compute_coverage(
    crop,
    threshold=THRESHOLD,
    min_artifact_area=MIN_ARTIFACT_AREA,
    method=METHOD,
    dilate=DILATE,
):
    """Return the fraction of food pixels in the cropped region."""
    mask = compute_mask(crop, threshold, min_artifact_area, method, dilate)
    food_pixels = int(np.count_nonzero(mask))
    total_pixels = mask.size
    if total_pixels == 0:
        return 0.0
    return food_pixels / total_pixels


def remove_small_artifacts(mask, min_area):
    """Remove connected components smaller than min_area from a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    # Label 0 is the background; skip it.
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def normalize_coverage(raw, minimum_coverage, full_coverage):
    """Remap raw coverage to 0-1 using the empty/full calibration bounds."""
    if full_coverage <= minimum_coverage:
        return 1.0 if raw >= full_coverage else 0.0
    normalized = (raw - minimum_coverage) / (full_coverage - minimum_coverage)
    return min(1.0, max(0.0, normalized))


def detect_image(
    image,
    roi=ROI,
    threshold=THRESHOLD,
    minimum_coverage=MINIMUM_COVERAGE,
    min_artifact_area=MIN_ARTIFACT_AREA,
    method=METHOD,
    dilate=DILATE,
    full_coverage=FULL_COVERAGE,
):
    """Run the detection pipeline on a decoded image (BGR numpy array)."""
    crop = crop_roi(image, roi)
    raw_coverage = compute_coverage(crop, threshold, min_artifact_area, method, dilate)
    coverage = normalize_coverage(raw_coverage, minimum_coverage, full_coverage)
    return {
        "food_present": raw_coverage >= minimum_coverage,
        "coverage": round(coverage, 2),
        "raw_coverage": round(raw_coverage, 2),
    }


def detect(
    path,
    roi=ROI,
    threshold=THRESHOLD,
    minimum_coverage=MINIMUM_COVERAGE,
    min_artifact_area=MIN_ARTIFACT_AREA,
    method=METHOD,
    dilate=DILATE,
    full_coverage=FULL_COVERAGE,
):
    """Run the full detection pipeline and return the result dictionary."""
    image = load_image(path)
    return detect_image(
        image,
        roi,
        threshold,
        minimum_coverage,
        min_artifact_area,
        method,
        dilate,
        full_coverage,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect cat food in a bowl snapshot.")
    parser.add_argument("image", help="Path to the snapshot image.")
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=ROI,
        help="Region of Interest as X Y WIDTH HEIGHT in pixels.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=THRESHOLD,
        help="Brightness threshold or, for texture, the Canny edge sensitivity.",
    )
    parser.add_argument(
        "--method",
        choices=("texture", "brightness"),
        default=METHOD,
        help="Detection method: texture (lighting-robust) or brightness.",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=DILATE,
        help="Dilation iterations for the texture method.",
    )
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=MINIMUM_COVERAGE,
        help="Raw coverage mapped to 0.0 (empty bowl floor).",
    )
    parser.add_argument(
        "--full-coverage",
        type=float,
        default=FULL_COVERAGE,
        help="Raw coverage mapped to 1.0 (full bowl ceiling).",
    )
    parser.add_argument(
        "--min-artifact-area",
        type=int,
        default=MIN_ARTIFACT_AREA,
        help="Minimum blob area (pixels) to keep; 0 disables artifact removal.",
    )
    args = parser.parse_args(argv)

    try:
        result = detect(
            args.image,
            roi=tuple(args.roi),
            threshold=args.threshold,
            minimum_coverage=args.minimum_coverage,
            min_artifact_area=args.min_artifact_area,
            method=args.method,
            dilate=args.dilate,
            full_coverage=args.full_coverage,
        )
    except (FileNotFoundError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
