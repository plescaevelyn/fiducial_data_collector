from __future__ import annotations
from typing import List, Dict, Any, Optional

from pathlib import Path

import cv2
import numpy as np


# Default template path for ChromaTag
_TEMPLATE_PATH = Path(__file__).with_name("chromatag_template.png")

_TEMPLATE_BGR: Optional[np.ndarray] = None
_TEMPLATE_HSV: Optional[np.ndarray] = None
_TEMPLATE_MASK: Optional[np.ndarray] = None
_TEMPLATE_W: int = 0
_TEMPLATE_H: int = 0


def _load_template() -> bool:
    """
    Load the ChromaTag template and build an HSV mask.

    The mask is used to focus template matching on the colored region,
    which is the main idea behind ChromaTag families.
    """
    global _TEMPLATE_BGR, _TEMPLATE_HSV, _TEMPLATE_MASK
    global _TEMPLATE_W, _TEMPLATE_H

    if _TEMPLATE_BGR is not None:
        return True

    if not _TEMPLATE_PATH.exists():
        print(f"[chromatag_backend] Template not found: {_TEMPLATE_PATH}")
        return False

    img = cv2.imread(str(_TEMPLATE_PATH), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[chromatag_backend] Failed to load template: {_TEMPLATE_PATH}")
        return False

    _TEMPLATE_BGR = img
    _TEMPLATE_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Build a broad color mask to emphasize chroma information.
    # You can tune these bounds for your specific marker design.
    # For now, we accept “strong saturation” pixels as marker region.
    h, s, v = cv2.split(_TEMPLATE_HSV)
    mask = (s > 40).astype(np.uint8) * 255
    _TEMPLATE_MASK = mask

    _TEMPLATE_H, _TEMPLATE_W = img.shape[:2]
    print(
        f"[chromatag_backend] Loaded template {str(_TEMPLATE_PATH)} "
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
    Naive ChromaTag detector using color-aware template matching.

    This is NOT the original ChromaTag algorithm, but a practical
    approximation that:

      - converts both template and input to HSV
      - uses the template's "high saturation" mask to emphasize colored areas
      - performs masked template matching in HSV->V (or luminance) channel

    Args:
        rgb_image: Input RGB image (H, W, 3).
        camera_matrix: Optional calibration (unused).
        dist_coeffs: Optional distortion coefficients (unused).
        depth_image: Optional depth map (unused).

    Returns:
        List with zero or one detections:
            {
              "marker_id": 0,
              "decoded": "chromatag_0",
              "confidence": <0..1>,
              "bbox": (x, y, w, h),
              "center": (cx, cy),
            }
    """
    if not _load_template():
        return []

    assert _TEMPLATE_BGR is not None
    assert _TEMPLATE_MASK is not None

    if rgb_image.ndim == 2:
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        # Assume BGR if coming from OpenCV, RGB if from somewhere else.
        # We heuristically treat it as BGR here; if your pipeline uses RGB,
        # just swap channels before calling detect().
        bgr = rgb_image

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_t, s_t, v_t = cv2.split(_TEMPLATE_HSV)  # type: ignore
    h_i, s_i, v_i = cv2.split(hsv)

    # Use V channel with the template mask for matching
    template = v_t
    mask = _TEMPLATE_MASK
    image = v_i

    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    THRESHOLD = 0.6

    if max_val < THRESHOLD:
        return []

    x, y = max_loc
    w, h = _TEMPLATE_W, _TEMPLATE_H   # type: ignore
    cx = x + w / 2.0
    cy = y + h / 2.0

    detection = {
        "marker_id": 0,
        "decoded": "chromatag_0",
        "confidence": float(max_val),
        "bbox": (int(x), int(y), int(w), int(h)),
        "center": (float(cx), float(cy)),
    }
    return [detection]
