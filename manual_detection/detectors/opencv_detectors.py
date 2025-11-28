# OpenCV-based fiducial marker detectors
# Implements detection for:
#   - aruco_4x4_50
#   - aruco_6x6_250
#   - apriltag_36h11  (via OpenCV's ArUco AprilTag dictionary)
#   - qr_code         (via cv2.QRCodeDetector)
#
# Expected result dictionary (used by DetectorManager + eval scripts):
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
        Input color image (H, W, 3), BGR or RGB.
    depth_image : np.ndarray
        Depth image (unused for 2D detection, kept for API compatibility).
    camera_matrix, dist_coeffs :
        Camera intrinsics (unused here but kept for future pose estimation).

    Returns
    -------
    Dict
        Standardized detection result dictionary.
    """
    # Convert to grayscale for all detectors
    if rgb_image.ndim == 3:
        # Works correctly whether input is RGB or BGR
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_image.copy()

    if marker_type == "qr_code":
        return _detect_qr(marker_type, gray)

    if marker_type in ("aruco_4x4_50", "aruco_6x6_250"):
        return _detect_aruco(marker_type, gray)

    if marker_type == "apriltag_36h11":
        return _detect_apriltag(marker_type, gray)

    # Safety fallback – normally never reached
    return {
        "detected": False,
        "marker_type": marker_type,
        "marker_id": None,
        "decoded": None,
        "confidence": None,
        "error": f"Unsupported OpenCV marker_type: {marker_type}",
    }


# =====================================================
# QR DETECTION (OpenCV QRCodeDetector)
# =====================================================

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

    # OpenCV QRCodeDetector provides no confidence score
    return {
        "detected": True,
        "marker_type": marker_type,
        "marker_id": None,        # QR codes have no numeric ID
        "decoded": decoded_text,  # QR payload
        "confidence": None,
        "error": None,
    }


# =====================================================
# ArUco / AprilTag DETECTION (via OpenCV ArUco module)
# =====================================================

def _get_aruco_module():
    """
    Helper to ensure OpenCV was built with opencv-contrib (aruco module).
    """
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

    # Support both new and old OpenCV ArUco APIs
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
        "confidence": None,  # OpenCV ArUco also provides no score
        "error": None,
    }


def _detect_apriltag(marker_type: str, gray: np.ndarray) -> Dict:
    """
    AprilTag 36h11 detection using OpenCV's built-in AprilTag dictionary.
    Does not require the external 'apriltag' library.
    """
    aruco = _get_aruco_module()

    try:
        dict_id = aruco.DICT_APRILTAG_36h11
    except AttributeError as e:
        return {
            "detected": False,
            "marker_type": marker_type,
            "marker_id": None,
            "decoded": None,
            "confidence": None,
            "error": f"OpenCV AprilTag dictionary not available: {e}",
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
