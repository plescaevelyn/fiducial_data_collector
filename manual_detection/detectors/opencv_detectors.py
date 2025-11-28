# OpenCV-based fiducial marker detectors
# Implements detection for:
#   - aruco_4x4_50
#   - aruco_6x6_250
#   - apriltag_36h11  (via OpenCV's ArUco AprilTag dictionary)
#   - qr_code         (via cv2.QRCodeDetector)
#
# Expected return dict (used by DetectorManager + eval scripts):
# {
#   "detected": bool,
#   "marker_type": str,
#   "marker_id": Optional[int],
#   "decoded": Optional[str],
#   "confidence": Optional[float],
#   "error": Optional[str],
# }

from typing import Dict, Optional

import cv2
import numpy as np


def detect_marker(
    marker_type: str,
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
) -> Dict:
    """
    Unified entry point for all OpenCV-based detectors.

    Parameters
    ----------
    marker_type : str
        One of: "aruco_4x4_50", "aruco_6x6_250", "apriltag_36h11", "qr_code".
    rgb_image : np.ndarray
        Input color image (H, W, 3), BGR or RGB – we'll just convert to gray.
    depth_image : np.ndarray
        Depth image (unused here, but kept for API compatibility).
    camera_matrix, dist_coeffs :
        Camera intrinsics (unused for 2D detection in synthetic dataset,
        but kept for potential pose estimation later).

    Returns
    -------
    Dict
        Detection result dict with keys:
        - detected (bool)
        - marker_type (str)
        - marker_id (Optional[int])
        - decoded (Optional[str])
        - confidence (Optional[float])
        - error (Optional[str])
    """
    # Convert to grayscale for all detectors
    if rgb_image.ndim == 3:
        # Works fine whether input is RGB or BGR
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_image.copy()

    if marker_type == "qr_code":
        return _detect_qr(marker_type, gray)

    if marker_type in ("aruco_4x4_50", "aruco_6x6_250"):
        return _detect_aruco(marker_type, gray)

    if marker_type == "apriltag_36h11":
        return _detect_apriltag(marker_type, gray)

    # Safety net – shouldn't normally get here, DetectorManager filtrează deja
    return {
        "detected": False,
        "marker_type": marker_type,
        "marker_id": None,
        "decoded": None,
        "confidence": None,
        "error": f"Unsupported OpenCV marker_type: {marker_type}",
    }


# =====================
# QR DETECTION (OpenCV)
# =====================


def _detect_qr(marker_type: str, gray: np.ndarray) -> Dict:
    qr = cv2.QRCodeDetector()

    try:
        decoded_text, points, _ = qr.detectAndDecode(gray)
    except Exception as e:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": f"QR detection error: {e}",
        }

    if points is None or decoded_text == "":
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": None,
        }

    # OpenCV QRCodeDetector nu dă un scor de încredere → confidence = None
    return {
        "detected": True,
        "marker_type": marker_type,
        "marker_id": None,           # QR nu are ID numeric, doar payload
        "decoded": decoded_text,     # conținutul QR-ului
        "confidence": None,
        "error": None,
    }


# ==========================
# ArUco / AprilTag (OpenCV)
# ==========================


def _get_aruco_module():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV was built without the 'aruco' module "
            "(opencv-contrib). Install opencv-contrib-python."
        )
    return cv2.aruco


def _detect_aruco(marker_type: str, gray: np.ndarray) -> Dict:
    aruco = _get_aruco_module()

    if marker_type == "aruco_4x4_50":
        dict_id = aruco.DICT_4X4_50
    elif marker_type == "aruco_6x6_250":
        dict_id = aruco.DICT_6X6_250
    else:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": f"Unknown ArUco marker_type: {marker_type}",
        }

    dictionary = aruco.getPredefinedDictionary(dict_id)

    # API compat: vechi (detectMarkers) vs nou (ArucoDetector)
    try:
        # OpenCV >= 4.7
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        # OpenCV mai vechi
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None or len(ids) == 0:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": None,
        }

    # Pentru dataset-ul tău sintetic avem un singur marker per imagine
    marker_id = int(ids[0][0])

    return {
        "detected": True,
        "marker_type": marker_type,
        "marker_id": marker_id,
        "decoded": None,    # ArUco nu are payload text
        "confidence": None, # OpenCV nu dă scor
        "error": None,
    }


def _detect_apriltag(marker_type: str, gray: np.ndarray) -> Dict:
    """
    AprilTag 36h11 prin OpenCV, folosind ArUco AprilTag dictionary.
    Nu mai depindem de librăria externă `apriltag`.
    """
    aruco = _get_aruco_module()

    # DICT_APRILTAG_36h11 este suportat de OpenCV cu contrib
    try:
        dict_id = aruco.DICT_APRILTAG_36h11
    except AttributeError as e:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": f"OpenCV ArUco AprilTag dictionary not available: {e}",
        }

    dictionary = aruco.getPredefinedDictionary(dict_id)

    try:
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None or len(ids) == 0:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": None,
        }

    marker_id = int(ids[0][0])

    return {
        "detected": True,
        "marker_type": marker_type,
        "marker_id": marker_id,
        "decoded": None,
        "confidence": None,
        "error": None,
    }
