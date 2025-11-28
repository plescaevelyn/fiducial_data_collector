# Centralized manager for all fiducial marker detectors
# Coordinates OpenCV and external detectors into a unified interface

import time
from typing import Dict, List, Optional

import cv2
import depthai as dai
import numpy as np

from .opencv_detectors import detect_marker as detect_opencv_marker
from .external_detectors import detect_external_marker


class DetectorManager:
    """
    Centralized manager for all fiducial marker detectors.
    Provides a unified interface for detecting all marker types.
    """

    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.is_calibrated = False

        self.opencv_markers = {
            "aruco_4x4_50",
            "aruco_6x6_250",
            "apriltag_36h11",
            "qr_code",
        }

        self.external_markers = {
            "runetag",
            "chromatag",
            "coppertag",
        }

        self.all_markers = self.opencv_markers | self.external_markers

        self.detection_stats = {
            marker: {"attempts": 0, "successes": 0, "avg_time": 0.0}
            for marker in self.all_markers
        }

    def calibrate_camera(self, device: dai.Device) -> bool:
        """
        Calibrates the OAK-D Lite camera for 3D pose estimation.
        """
        try:
            calib_data = device.readCalibration()
            self.camera_matrix = np.array(
                calib_data.getCameraIntrinsics(
                    dai.CameraBoardSocket.RGB, 640, 480
                )
            )
            self.dist_coeffs = np.array(
                calib_data.getDistortionCoefficients(
                    dai.CameraBoardSocket.RGB
                )
            )
            self.is_calibrated = True

            print("Camera successfully calibrated")
            return True

        except Exception as e:
            print(f"Camera calibration error: {e}")
            self.is_calibrated = False
            return False

    def detect_marker(
        self,
        marker_type: str,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Detects a specific marker using the appropriate detector.

        depth_image can be None (e.g. synthetic dataset). In that case
        a dummy depth map filled with zeros is used.
        """
        if depth_image is None:
            depth_image = np.zeros(rgb_image.shape[:2], dtype=np.uint16)

        if marker_type not in self.all_markers:
            return {
                "detected": False,
                "marker_type": marker_type,
                "error": f"Marker type {marker_type} not supported",
            }

        self.detection_stats[marker_type]["attempts"] += 1

        start_time = time.perf_counter()

        try:
            if marker_type in self.opencv_markers:
                result = detect_opencv_marker(
                    marker_type,
                    rgb_image,
                    depth_image,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            elif marker_type in self.external_markers:
                result = detect_external_marker(
                    marker_type,
                    rgb_image,
                    depth_image,
                    self.camera_matrix,
                    self.dist_coeffs,
                )
            else:
                result = {
                    "detected": False,
                    "marker_type": marker_type,
                    "error": "Unknown marker category",
                }

            total_time = (time.perf_counter() - start_time) * 1000.0

            result["total_detection_time_ms"] = total_time
            result["calibration_available"] = self.is_calibrated
            result["timestamp"] = time.time()

            if result.get("detected", False):
                self.detection_stats[marker_type]["successes"] += 1

            stats = self.detection_stats[marker_type]
            stats["avg_time"] = (
                stats["avg_time"] * (stats["attempts"] - 1) + total_time
            ) / stats["attempts"]

            return result

        except Exception as e:
            return {
                "detected": False,
                "marker_type": marker_type,
                "error": f"Detection failed: {str(e)}",
                "total_detection_time_ms": (time.perf_counter() - start_time)
                * 1000.0,
            }

    def detect_all_markers(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
    ) -> Dict[str, Dict]:
        """
        Detects all marker types in the given image.
        Useful for comparisons and evaluations.
        """
        results: Dict[str, Dict] = {}
        print("Detecting all marker types...")
        for marker_type in self.all_markers:
            result = self.detect_marker(marker_type, rgb_image, depth_image)
            results[marker_type] = result
        return results

    def get_detection_statistics(self) -> Dict:
        """
        Returns detection statistics for all markers.
        """
        stats_out: Dict[str, Dict] = {}
        for marker_type, data in self.detection_stats.items():
            if data["attempts"] > 0:
                success_rate = (data["successes"] / data["attempts"]) * 100.0
                stats_out[marker_type] = {
                    "attempts": data["attempts"],
                    "successes": data["successes"],
                    "success_rate_percent": success_rate,
                    "avg_detection_time_ms": data["avg_time"],
                }
            else:
                stats_out[marker_type] = {
                    "attempts": 0,
                    "successes": 0,
                    "success_rate_percent": 0.0,
                    "avg_detection_time_ms": 0.0,
                }
        return stats_out

    def print_detection_summary(self):
        """
        Displays a summary of detector performance.
        """
        stats = self.get_detection_statistics()
        print("\nDETECTION PERFORMANCE SUMMARY")
        print("=" * 60)
        print(
            f"{'Marker':<15} {'Attempts':<10} {'Successes':<10} "
            f"{'Success Rate':<12} {'Avg Time':<12}"
        )
        print("-" * 60)
        for marker_type, data in stats.items():
            print(
                f"{marker_type:<15} "
                f"{data['attempts']:<10} "
                f"{data['successes']:<10} "
                f"{data['success_rate_percent']:<11.1f}% "
                f"{data['avg_detection_time_ms']:<11.1f}ms"
            )

    def validate_marker_availability(self) -> Dict[str, bool]:
        """
        Validates availability of all detectors.
        """
        availability: Dict[str, bool] = {}
        print("Validating detector availability...")

        for marker_type in self.opencv_markers:
            try:
                test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                test_depth = np.zeros((480, 640), dtype=np.uint16)
                detect_opencv_marker(
                    marker_type, test_image, test_depth, None, None
                )
                availability[marker_type] = True
            except Exception:
                availability[marker_type] = False

        for marker_type in self.external_markers:
            try:
                test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                test_depth = np.zeros((480, 640), dtype=np.uint16)
                detect_external_marker(
                    marker_type, test_image, test_depth, None, None
                )
                availability[marker_type] = True
            except Exception:
                availability[marker_type] = False

        return availability

    def get_recommended_test_order(self) -> List[str]:
        """
        Returns recommended test order based on success rate and performance.
        """
        stats = self.get_detection_statistics()
        sorted_markers = sorted(
            self.all_markers,
            key=lambda m: (
                stats[m]["success_rate_percent"],
                -stats[m]["avg_detection_time_ms"],
            ),
            reverse=True,
        )
        return sorted_markers

    def reset_statistics(self):
        """
        Resets detection statistics.
        """
        self.detection_stats = {
            marker: {"attempts": 0, "successes": 0, "avg_time": 0.0}
            for marker in self.all_markers
        }
        print("Detection statistics reset")


def create_detector_manager(oak_device: Optional[dai.Device] = None) -> DetectorManager:
    """
    Creates and initializes a DetectorManager.
    """
    manager = DetectorManager()
    if oak_device is not None:
        availability = manager.validate_marker_availability()
        print("Detector availability:", availability)
    else:
        manager.validate_marker_availability()
    return manager
