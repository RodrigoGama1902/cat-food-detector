"""ROI helper for the Cat Food Detector.

Opens an image in a window so you can draw a rectangle with the mouse.
On confirmation it prints the ROI as (x, y, width, height), ready to paste
into detector.py or to pass via --roi.

Usage:
    python roi_helper.py latest.jpg

Controls:
    - Drag with the mouse to draw the rectangle.
    - ENTER or SPACE to confirm the selection.
    - C or ESC to cancel.
"""

import argparse
import sys

import cv2

# Largest dimension (pixels) the preview window is allowed to use.
MAX_DISPLAY_SIZE = 1000


def select_roi(path):
    """Open the image and let the user draw a ROI. Returns (x, y, w, h) or None."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    height, width = image.shape[:2]
    # Scale the image down for display if it is larger than MAX_DISPLAY_SIZE.
    scale = min(1.0, MAX_DISPLAY_SIZE / max(width, height))
    if scale < 1.0:
        display = cv2.resize(image, (round(width * scale), round(height * scale)))
    else:
        display = image

    window = "Select ROI - drag, ENTER to confirm, C/ESC to cancel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display.shape[1], display.shape[0])
    roi = cv2.selectROI(window, display, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None

    # Map the selection back to the original image resolution.
    return (
        round(x / scale),
        round(y / scale),
        round(w / scale),
        round(h / scale),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interactively pick a ROI from an image.")
    parser.add_argument("image", help="Path to the snapshot image.")
    args = parser.parse_args(argv)

    try:
        roi = select_roi(args.image)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    if roi is None:
        print("No ROI selected.", file=sys.stderr)
        return 1

    x, y, w, h = roi
    print("ROI selected:")
    print(f"  Tuple (for detector.py):  ROI = ({x}, {y}, {w}, {h})")
    print(f"  CLI flag:                 --roi {x} {y} {w} {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
