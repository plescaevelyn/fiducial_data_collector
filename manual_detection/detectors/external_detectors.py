"""
External (non-OpenCV) fiducial marker detectors.

This module provides a unified adapter API for:
  - RuneTag
  - ChromaTag
  - CopperTag

The DetectorManager can treat all detectors uniformly via
`detect_external_marker(...)`.

Real detection logic is expected to live in separate backend modules
that you provide, e.g.:

  - runetag_backend
  - chromatag_backend
  - coppertag_backend

Each backend should expose a `detect(...)` function that receives the
RGB image (and optionally calibration / depth) and returns a list of
detections.
"""

from typing import Dict, Optional, List, Any

import numpy as np

# Optional backends; imported lazily and handled gracefully if missing.
try:
    import runetag_backend  # type: ignore
except ImportError:
    runetag_backend = None  # type: ignore

try:
    import chromatag_backend  # type: ignore
except ImportError:
    chromatag_backend = None  # type: ignore

try:
    import coppertag_backend  # type: ignore
except ImportError:
    coppertag_backend = None  # type: ignore


def _format_no_backend_result(marker_type: str) -> Dict:
    """
    Build a consistent "backend missing" result payload.
    """
    return {
        "detected": False,
        "marker_type": marker_type,
        "marker_id": None,
        "decoded": None,
        "confidence": None,
        "error": (
            f"External detector backend for '{marker_type}' is not available "
            f"(backend module not imported)"
        ),
    }


def _format_detection_result(
    marker_type: str,
    detections: List[Dict[str, Any]],
) -> Dict:
    """
    Convert a list of backend detections into the standardized
    result dictionary expected by DetectorManager.

    Convention used here:
      - If no detections -> detected = False
      - If there are detections, the first one is used as the primary
        detection; the rest are ignored at this level.
      - Backend detection dict is expected to contain some of:
          * "id" or "marker_id"
          * "decoded" or "text"
          * "confidence" or "score"

    You can adapt this helper to match your actual backend output.
    """
    if not detections:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": None,
        }

    det = detections[0]

    marker_id = det.get("marker_id", det.get("id"))
    decoded = det.get("decoded", det.get("text"))
    confidence = det.get("confidence", det.get("score"))

    return {
        "detected": True,
        "marker_type": marker_type,
        "marker_id": marker_id,
        "decoded": decoded,
        "confidence": confidence,
        "error": None,
    }


def _detect_runetag(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
) -> Dict:
    """
    RuneTag detection adapter.

    Expected backend: `runetag_backend.detect(...)`.

    A suggested backend signature is:

        detections = runetag_backend.detect(
            rgb_image,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            depth_image=depth_image,
        )

    where `detections` is a list of dicts.

    You can adapt this call to your actual RuneTag API.
    """
    marker_type = "runetag"

    if runetag_backend is None:
        return _format_no_backend_result(marker_type)

    try:
        detections = runetag_backend.detect(
            rgb_image,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            depth_image=depth_image,
        )
        if not isinstance(detections, list):
            detections = []
        return _format_detection_result(marker_type, detections)
    except Exception as e:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": f"RuneTag detection failed: {e}",
        }


def _detect_chromatag(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
) -> Dict:
    """
    ChromaTag detection adapter.

    Expected backend: `chromatag_backend.detect(...)`.

    Suggested backend signature:

        detections = chromatag_backend.detect(
            rgb_image,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            depth_image=depth_image,
        )

    Again, `detections` should be a list of dicts.
    """
    marker_type = "chromatag"

    if chromatag_backend is None:
        return _format_no_backend_result(marker_type)

    try:
        detections = chromatag_backend.detect(
            rgb_image,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            depth_image=depth_image,
        )
        if not isinstance(detections, list):
            detections = []
        return _format_detection_result(marker_type, detections)
    except Exception as e:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": f"ChromaTag detection failed: {e}",
        }


def _detect_coppertag(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
) -> Dict:
    """
    CopperTag detection adapter.

    Expected backend: `coppertag_backend.detect(...)`.

    Suggested backend signature:

        detections = coppertag_backend.detect(
            rgb_image,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            depth_image=depth_image,
        )

    `detections` should be a list of dicts compatible with
    `_format_detection_result`.
    """
    marker_type = "coppertag"

    if coppertag_backend is None:
        return _format_no_backend_result(marker_type)

    try:
        detections = coppertag_backend.detect(
            rgb_image,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            depth_image=depth_image,
        )
        if not isinstance(detections, list):
            detections = []
        return _format_detection_result(marker_type, detections)
    except Exception as e:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": f"CopperTag detection failed: {e}",
        }


def detect_external_marker(
    marker_type: str,
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
) -> Dict:
    """
    Entry point for external detectors (RuneTag / ChromaTag / CopperTag).

    This function dispatches to the appropriate adapter and returns a
    unified result dictionary compatible with DetectorManager.

    If the backend for a given marker family is not available, a
    graceful error message is returned instead of raising.
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

    if marker_type == "runetag":
        return _detect_runetag(
            rgb_image,
            depth_image,
            camera_matrix,
            dist_coeffs,
        )
    if marker_type == "chromatag":
        return _detect_chromatag(
            rgb_image,
            depth_image,
            camera_matrix,
            dist_coeffs,
        )
    if marker_type == "coppertag":
        return _detect_coppertag(
            rgb_image,
            depth_image,
            camera_matrix,
            dist_coeffs,
        )

    # Fallback (should not be reached)
    return {
        "detected": False,
        "marker_type": marker_type,
        "marker_id": None,
        "decoded": None,
        "confidence": None,
        "error": "Unhandled external marker type in dispatcher",
    }
