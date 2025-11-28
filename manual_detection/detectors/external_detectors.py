# External (non-OpenCV) fiducial marker detectors
# Placeholder implementations for:
#   - runetag
#   - chromatag
#   - coppertag
#
# The API matches the OpenCV detectors so DetectorManager can treat all
# detectors uniformly.

from typing import Dict, Optional

import numpy as np


def detect_external_marker(
    marker_type: str,
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
) -> Dict:
    """
    Entry point for external detectors (RuneTag / ChromaTag / CopperTag).

    For now these are NOT implemented and always report:
      detected = False
      error = "External detector for <marker_type> not implemented"

    This avoids AttributeError in DetectorManager and makes it explicit
    in the JSON why detection failed.
    """
    supported = {"runetag", "chromatag", "coppertag"}

    if marker_type not in supported:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": f"Unknown external marker_type: {marker_type}",
        }

    # Când vei implementa detectorii reali, înlocuiești bucata asta cu:
    # if marker_type == "runetag": ...
    # if marker_type == "chromatag": ...
    # if marker_type == "coppertag": ...
    return {
        "detected": False,
        "marker_type": marker_type,
        "marker_id": None,
        "decoded": None,
        "confidence": None,
        "error": f"External detector for '{marker_type}' not implemented",
    }
