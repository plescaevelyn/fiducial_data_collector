from __future__ import annotations
from typing import List, Dict, Any, Optional

from pathlib import Path

import cv2
import numpy as np


# Default template path for CopperTag
_TEMPLATE_PATH = Path(__file__).with_name("coppertag_template.png")

_TEMPLATE_GRAY: Optional[np.ndarray] = None
_TEMPLATE_EDGES: Optional[np.ndarray] = None
_TEMPLATE_W: int = 0
_TEMPLATE_H: int = 0


def _load_template() -> bool:
    """
    Load the CopperTag template and precompute its edge map.

    CopperTag uses special patterns that often have strong edge structure;
    this naive detector emphasizes edges for matching.
    """
    global _TEMPLATE_GRAY, _TEMPLATE_EDGES, _TEMPLATE_W, _TEMPLATE_H

    if _TEMPLATE_GRAY is not None:
        return True

    if not _TEMPLATE_PATH.exists():
        print(f"[coppertag_backend] Template not found: {_TEMPLATE_PATH}")
        return False

    img = cv2.imread(str(_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[coppertag_backend] Failed to load template: {_TEMPLATE_PATH}")
        return False

    _TEMPLATE_GRAY = img
    _TEMPLATE_EDGES = cv2.Canny(img, 50, 150)
    _TEMPLATE_H, _TEMPLATE_W = img.shape[:2]

    print(
        f"[coppertag_backend] Loaded template {str(_TEMPLATE_PATH)} "
        f"({ _TEMPLATE_W }x{ _TEMPLATE_H })"
    )
    return True


def detect(
    rgb_image: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    depth_image: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Naive CopperTag detector using edge-based template matching.

    This is NOT the official CopperTag detector, but a practical
    approach for experiments:

      - converts the input to grayscale
      - runs Canny edge detection
      - performs template matching in edge space

    Args:
        rgb_image: Input RGB image (H, W, 3).
        camera_matrix: Optional calibration (unused).
        dist_coeffs: Optional distortion coefficients (unused).
        depth_image: Optional depth map (unused).

    Returns:
        List with zero or one detections:
            {
              "marker_id": 0,
              "decoded": "coppertag_0",
              "confidence": <0..1>,
              "bbox": (x, y, w, h),
              "center": (cx, cy),
            }
    """
    if not _load_template():
        return []

    assert _TEMPLATE_EDGES is not None
    template_edges = _TEMPLATE_EDGES

    # Convert input to grayscale, then edges
    if rgb_image.ndim == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_image.copy()

    img_edges = cv2.Canny(gray, 50, 150)

    res = cv2.matchTemplate(img_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    THRESHOLD = 0.5  # edge-based matching is noisier; threshold slightly lower

    if max_val < THRESHOLD:
        return []

    x, y = max_loc
    w, h = _TEMPLATE_W, _TEMPLATE_H  # type: ignore
    cx = x + w / 2.0
    cy = y + h / 2.0

    detection = {
        "marker_id": 0,
        "decoded": "coppertag_0",
        "confidence": float(max_val),
        "bbox": (int(x), int(y), int(w), int(h)),
        "center": (float(cx), float(cy)),
    }
    return [detection]
