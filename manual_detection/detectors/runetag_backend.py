from __future__ import annotations
from typing import List, Dict, Any, Optional

from pathlib import Path

import cv2
import numpy as np


# Default template path: same folder as this file
_TEMPLATE_PATH = Path(__file__).with_name("runetag_template.png")

# Global cache so we do not reload template on every call
_TEMPLATE_GRAY: Optional[np.ndarray] = None
_TEMPLATE_W: int = 0
_TEMPLATE_H: int = 0


def _load_template() -> Optional[np.ndarray]:
    """
    Loads the RuneTag template image from disk (once) and converts it to gray.

    Returns:
        Grayscale template or None if loading failed.
    """
    global _TEMPLATE_GRAY, _TEMPLATE_W, _TEMPLATE_H

    if _TEMPLATE_GRAY is not None:
        return _TEMPLATE_GRAY

    if not _TEMPLATE_PATH.exists():
        print(f"[runetag_backend] Template not found: {_TEMPLATE_PATH}")
        return None

    img = cv2.imread(str(_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[runetag_backend] Failed to load template: {_TEMPLATE_PATH}")
        return None

    _TEMPLATE_GRAY = img
    _TEMPLATE_H, _TEMPLATE_W = img.shape[:2]
    print(
        f"[runetag_backend] Loaded template {str(_TEMPLATE_PATH)} "
        f"({ _TEMPLATE_W }x{ _TEMPLATE_H })"
    )
    return _TEMPLATE_GRAY


def detect(
    rgb_image: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    depth_image: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Naive RuneTag detector based on template matching.

    This is NOT the official RuneTag algorithm. It is a practical,
    reproducible detector that:

      - converts the input to grayscale
      - performs cv2.matchTemplate with a reference RuneTag image
      - if the best match exceeds a threshold, reports one detection

    Args:
        rgb_image: Input RGB image (H, W, 3).
        camera_matrix: Optional calibration (unused here).
        dist_coeffs: Optional distortion coefficients (unused here).
        depth_image: Optional depth map (unused here).

    Returns:
        List with zero or one detection dictionaries:
            {
              "marker_id": 0,
              "decoded": "runetag_0",
              "confidence": <0..1>,
              "bbox": (x, y, w, h),
              "center": (cx, cy),
            }
    """
    template = _load_template()
    if template is None:
        # No template available -> no detections
        return []

    # Convert input to grayscale
    if rgb_image.ndim == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_image.copy()

    th, tw = template.shape[:2]

    # Template matching (normalized correlation)
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Set a conservative threshold; you can tune this if you know your data
    THRESHOLD = 0.6

    if max_val < THRESHOLD:
        return []

    x, y = max_loc
    w, h = tw, th
    cx = x + w / 2.0
    cy = y + h / 2.0

    detection = {
        "marker_id": 0,             # Single canonical RuneTag for now
        "decoded": "runetag_0",
        "confidence": float(max_val),
        "bbox": (int(x), int(y), int(w), int(h)),
        "center": (float(cx), float(cy)),
    }
    return [detection]
