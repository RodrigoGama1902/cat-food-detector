"""Flask API + configuration UI for the Cat Food Detector.

Endpoints:
    GET  /                       Configuration UI.
    GET  /health                 Liveness probe.
    POST /detect                 Run detection on an image. Use ?night_mode=true
                                 to apply the night profile (default: day).
    GET  /api/config             Current saved configuration.
    POST /api/roi                Persist the ROI ([x, y, w, h]).
    POST /api/calibrate/upload   Upload empty/medium/full images for a profile.
    GET  /api/calibrate/preview  Live detection preview for a profile's params.
    POST /api/calibrate/save     Persist a profile's threshold/method/etc.

Designed to run inside a container; mount a volume for config.json so the
calibration persists across restarts (CONFIG_PATH env var).
"""

import base64
import os
import tempfile

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from config import (
    PROFILE_NAMES,
    apply_config,
    get_profile,
    load_config,
    save_config,
    save_profile,
)
from detector import compute_mask, crop_roi, detect_image, normalize_coverage

app = Flask(__name__)

# Calibration images are stored on disk so they survive across gunicorn
# workers within the container. They are intentionally ephemeral.
CALIB_DIR = os.environ.get(
    "CALIB_DIR", os.path.join(tempfile.gettempdir(), "cat_food_calib")
)
os.makedirs(CALIB_DIR, exist_ok=True)

SLOTS = ("empty", "medium", "full")

_TRUE = {"1", "true", "yes", "on"}

# Mean saturation below this means the frame is effectively grayscale (IR night
# mode). Color daytime frames sit well above it.
NIGHT_SATURATION_MAX = 12.0


def _is_night(value, default=False):
    """Parse a night_mode flag from a string value."""
    if value is None:
        return default
    return str(value).strip().lower() in _TRUE


def detect_night_image(image):
    """Return True if the frame looks like night mode (near-grayscale / IR)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = float(hsv[:, :, 1].mean())
    return mean_saturation < NIGHT_SATURATION_MAX, mean_saturation


def _profile_name(value):
    """Return a valid profile name, defaulting to 'day'."""
    return value if value in PROFILE_NAMES else "day"


def _read_image_bytes():
    """Return the raw image bytes from the request, or None if absent."""
    if "image" in request.files:
        return request.files["image"].read()
    if request.data:
        return request.data
    return None


def _png_b64(image):
    """Encode an image (BGR or grayscale) as a base64 data URL."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def _calib_path(profile, slot):
    return os.path.join(CALIB_DIR, f"{profile}_{slot}.img")


# --- UI ----------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


# --- Detection ---------------------------------------------------------------
@app.post("/detect")
def detect():
    raw = _read_image_bytes()
    if not raw:
        return jsonify({"error": "No image provided."}), 400

    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image."}), 400

    config = load_config()
    args = request.args
    # The detector auto-detects day/night from the image (grayscale = night).
    # An explicit ?night_mode=... query param overrides the auto-detection.
    auto_night, mean_saturation = detect_night_image(image)
    override = args.get("night_mode")
    night_mode = _is_night(override) if override is not None else auto_night
    profile = get_profile(config, night_mode)
    roi = (
        args.get("x", config["roi"][0], type=int),
        args.get("y", config["roi"][1], type=int),
        args.get("w", config["roi"][2], type=int),
        args.get("h", config["roi"][3], type=int),
    )

    try:
        result = detect_image(
            image,
            roi=roi,
            threshold=args.get("threshold", profile["threshold"], type=int),
            minimum_coverage=args.get(
                "minimum_coverage", profile["minimum_coverage"], type=float
            ),
            min_artifact_area=args.get(
                "min_artifact_area", config["min_artifact_area"], type=int
            ),
            method=args.get("method", profile["method"]),
            dilate=args.get("dilate", profile["dilate"], type=int),
            full_coverage=args.get(
                "full_coverage", profile["full_coverage"], type=float
            ),
            cluster_k=args.get("cluster_k", profile.get("cluster_k", 4), type=int),
            cluster_min_texture=args.get(
                "cluster_min_texture",
                profile.get("cluster_min_texture", 0.08),
                type=float,
            ),
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    result["night_mode"] = night_mode
    result["auto_detected"] = override is None
    result["mean_saturation"] = round(mean_saturation, 1)
    return jsonify(result)


# --- Configuration API -------------------------------------------------------
@app.get("/api/config")
def get_config():
    return jsonify(load_config())


@app.post("/api/config")
def update_config():
    """Persist a full config edited from the header text field.

    Accepts {"roi": [...], "profiles": {"day": {...}, "night": {...}}}; any
    omitted keys keep their saved values. Creates the config file if absent.
    """
    data = request.get_json(silent=True) or {}
    updates = {}

    roi = data.get("roi")
    if roi is not None:
        if len(roi) != 4:
            return jsonify({"error": "roi must be [x, y, w, h]"}), 400
        try:
            roi = [int(v) for v in roi]
        except (TypeError, ValueError):
            return jsonify({"error": "roi values must be integers"}), 400
        if roi[2] <= 0 or roi[3] <= 0:
            return jsonify({"error": "width and height must be positive"}), 400
        updates["roi"] = roi

    profiles = data.get("profiles") or {}
    clean_profiles = {}
    for name, raw in profiles.items():
        if name not in PROFILE_NAMES or not isinstance(raw, dict):
            continue
        profile = {}
        try:
            if "method" in raw:
                if raw["method"] not in ("texture", "brightness", "cluster"):
                    return jsonify({"error": f"invalid method: {raw['method']}"}), 400
                profile["method"] = raw["method"]
            if "threshold" in raw:
                profile["threshold"] = int(raw["threshold"])
            if "minimum_coverage" in raw:
                profile["minimum_coverage"] = float(raw["minimum_coverage"])
            if "full_coverage" in raw:
                profile["full_coverage"] = float(raw["full_coverage"])
            if "dilate" in raw:
                profile["dilate"] = int(raw["dilate"])
            if "cluster_k" in raw:
                profile["cluster_k"] = int(raw["cluster_k"])
            if "cluster_min_texture" in raw:
                profile["cluster_min_texture"] = float(raw["cluster_min_texture"])
        except (TypeError, ValueError):
            return jsonify({"error": f"invalid values for profile '{name}'"}), 400
        if profile:
            clean_profiles[name] = profile
    if clean_profiles:
        updates["profiles"] = clean_profiles

    if not updates:
        return jsonify({"error": "nothing to update"}), 400
    return jsonify(apply_config(updates))


@app.post("/api/roi")
def set_roi():
    data = request.get_json(silent=True) or {}
    roi = data.get("roi")
    if not roi or len(roi) != 4:
        return jsonify({"error": "roi must be [x, y, w, h]"}), 400
    try:
        roi = [int(v) for v in roi]
    except (TypeError, ValueError):
        return jsonify({"error": "roi values must be integers"}), 400
    if roi[2] <= 0 or roi[3] <= 0:
        return jsonify({"error": "width and height must be positive"}), 400
    return jsonify(save_config({"roi": roi}))


@app.post("/api/calibrate/upload")
def calibrate_upload():
    profile = _profile_name(request.args.get("profile") or request.form.get("profile"))
    saved = []
    for slot in SLOTS:
        file = request.files.get(slot)
        if file and file.filename:
            file.save(_calib_path(profile, slot))
            saved.append(slot)
    if not saved:
        return jsonify({"error": "no images provided"}), 400
    return jsonify({"profile": profile, "saved": saved})


@app.get("/api/calibrate/preview")
def calibrate_preview():
    config = load_config()
    roi = tuple(config["roi"])
    profile_name = _profile_name(request.args.get("profile"))
    profile = config["profiles"][profile_name]
    threshold = request.args.get("threshold", profile["threshold"], type=int)
    minimum_coverage = request.args.get(
        "minimum_coverage", profile["minimum_coverage"], type=float
    )
    full_coverage = request.args.get(
        "full_coverage", profile["full_coverage"], type=float
    )
    method = request.args.get("method", profile["method"])
    dilate = request.args.get("dilate", profile["dilate"], type=int)
    cluster_k = request.args.get("cluster_k", profile.get("cluster_k", 4), type=int)
    cluster_min_texture = request.args.get(
        "cluster_min_texture", profile.get("cluster_min_texture", 0.08), type=float
    )
    min_artifact_area = config["min_artifact_area"]

    results = {}
    for slot in SLOTS:
        path = _calib_path(profile_name, slot)
        if not os.path.exists(path):
            continue
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        try:
            crop = crop_roi(image, roi)
        except ValueError as error:
            results[slot] = {"error": str(error)}
            continue
        mask = compute_mask(
            crop, threshold, min_artifact_area, method, dilate, cluster_k, cluster_min_texture
        )
        coverage = (
            round(float(np.count_nonzero(mask)) / mask.size, 2) if mask.size else 0.0
        )
        results[slot] = {
            "coverage": coverage,
            "normalized": round(
                normalize_coverage(coverage, minimum_coverage, full_coverage), 2
            ),
            "food_present": coverage >= minimum_coverage,
            "crop": _png_b64(crop),
            "mask": _png_b64(mask),
        }

    return jsonify(
        {
            "profile": profile_name,
            "threshold": threshold,
            "minimum_coverage": minimum_coverage,
            "full_coverage": full_coverage,
            "method": method,
            "dilate": dilate,
            "cluster_k": cluster_k,
            "cluster_min_texture": cluster_min_texture,
            "results": results,
        }
    )


@app.post("/api/calibrate/save")
def calibrate_save():
    data = request.get_json(silent=True) or {}
    profile_name = _profile_name(data.get("profile"))
    updates = {}
    if "threshold" in data:
        updates["threshold"] = int(data["threshold"])
    if "minimum_coverage" in data:
        updates["minimum_coverage"] = float(data["minimum_coverage"])
    if "full_coverage" in data:
        updates["full_coverage"] = float(data["full_coverage"])
    if "method" in data and data["method"] in ("texture", "brightness", "cluster"):
        updates["method"] = data["method"]
    if "dilate" in data:
        updates["dilate"] = int(data["dilate"])
    if "cluster_k" in data:
        updates["cluster_k"] = int(data["cluster_k"])
    if "cluster_min_texture" in data:
        updates["cluster_min_texture"] = float(data["cluster_min_texture"])
    if not updates:
        return jsonify({"error": "nothing to save"}), 400
    return jsonify(save_profile(profile_name, updates))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
