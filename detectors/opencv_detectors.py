# OpenCV detectors for standard fiducial markers
# Implements ArUco, AprilTag, and QR Code using native OpenCV

import cv2
import numpy as np
from typing import Dict, Optional
import time

class OpenCVDetectors:
    """
    Detectors for markers available natively in OpenCV.
    Implements ArUco, AprilTag, and QR Code with full metrics.
    """

    def __init__(self):
        self.aruco_dicts = {
            'aruco_4x4_50': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            'aruco_6x6_250': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
            'apriltag_36h11': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        }

        self.aruco_params = cv2.aruco.DetectorParameters()
        self.qr_detector = cv2.QRCodeDetector()

        self.camera_matrix = None
        self.dist_coeffs = None
        self.marker_size = 0.05

    def set_camera_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def detect_aruco_4x4_50(self, rgb_image, depth_image):
        return self._detect_aruco_generic(rgb_image, depth_image, 'aruco_4x4_50')

    def detect_aruco_6x6_250(self, rgb_image, depth_image):
        return self._detect_aruco_generic(rgb_image, depth_image, 'aruco_6x6_250')

    def detect_apriltag_36h11(self, rgb_image, depth_image):
        return self._detect_aruco_generic(rgb_image, depth_image, 'apriltag_36h11')

    def detect_qr_code(self, rgb_image, depth_image):
        start_time = time.perf_counter()
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(gray)
        detection_time = (time.perf_counter() - start_time) * 1000

        if retval and points is not None and len(points) > 0:
            corners = points[0].reshape(4, 2).astype(np.float32)
            data = decoded_info[0] if decoded_info else ""
            pose_3d = self._estimate_3d_pose(corners)
            metrics = self._calculate_detection_metrics(corners, depth_image, detection_time)
            return {
                'detected': True,
                'marker_type': 'qr_code',
                'corners': corners.tolist(),
                'data': data,
                'pose_3d': pose_3d,
                'metrics': metrics,
                'detection_method': 'opencv_qr'
            }

        return {
            'detected': False,
            'marker_type': 'qr_code',
            'detection_method': 'opencv_qr',
            'detection_time_ms': detection_time
        }

    def _detect_aruco_generic(self, rgb_image, depth_image, marker_type):
        start_time = time.perf_counter()
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        dictionary = self.aruco_dicts[marker_type]
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=self.aruco_params)
        detection_time = (time.perf_counter() - start_time) * 1000

        if ids is not None and len(ids) > 0:
            marker_corners = corners[0][0].astype(np.float32)
            marker_id = int(ids[0][0])
            pose_3d = self._estimate_3d_pose(marker_corners)
            metrics = self._calculate_detection_metrics(marker_corners, depth_image, detection_time)
            return {
                'detected': True,
                'marker_type': marker_type,
                'id': marker_id,
                'corners': marker_corners.tolist(),
                'pose_3d': pose_3d,
                'metrics': metrics,
                'detection_method': 'opencv_aruco'
            }

        return {
            'detected': False,
            'marker_type': marker_type,
            'detection_method': 'opencv_aruco',
            'detection_time_ms': detection_time
        }

    def _estimate_3d_pose(self, corners: np.ndarray) -> Optional[Dict]:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return None

        half_size = self.marker_size / 2.0
        object_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(object_points, corners, self.camera_matrix, self.dist_coeffs)
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            return {
                'position': tvec.flatten().tolist(),
                'rotation_vector': rvec.flatten().tolist(),
                'rotation_matrix': rotation_matrix.tolist(),
                'distance': float(np.linalg.norm(tvec))
            }
        return None

    def _calculate_detection_metrics(self, corners: np.ndarray, depth_image: np.ndarray, detection_time: float) -> Dict:
        metrics = {
            'detection_time_ms': detection_time,
            'corners_detected': len(corners),
            'corners_expected': 4,
            'corner_accuracy': len(corners) / 4.0
        }

        if len(corners) == 4:
            area = cv2.contourArea(corners)
            perimeter = cv2.arcLength(corners, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
                metrics['shape_compactness'] = float(compactness)

            center = np.mean(corners, axis=0)
            metrics['center_pixel'] = center.tolist()

            if depth_image is not None:
                cx, cy = int(center[0]), int(center[1])
                if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                    depth_value = depth_image[cy, cx]
                    if depth_value > 0:
                        metrics['depth_mm'] = float(depth_value)
                        metrics['depth_m'] = float(depth_value / 1000.0)
                        metrics['depth_quality'] = self._assess_depth_quality(corners, depth_image)
        return metrics

    def _assess_depth_quality(self, corners: np.ndarray, depth_image: np.ndarray) -> float:
        if depth_image is None or len(corners) != 4:
            return 0.0

        mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [corners.astype(np.int32)], 255)

        depth_values = depth_image[mask > 0]
        valid_depths = depth_values[depth_values > 0]

        if len(valid_depths) == 0:
            return 0.0

        validity_ratio = len(valid_depths) / len(depth_values)
        if len(valid_depths) > 10:
            depth_std = np.std(valid_depths)
            depth_mean = np.mean(valid_depths)
            uniformity = 1.0 - min(1.0, depth_std / depth_mean) if depth_mean > 0 else 0.0
        else:
            uniformity = 0.0

        quality_score = validity_ratio * 0.6 + uniformity * 0.4
        return float(quality_score)


def detect_marker(marker_type: str, rgb_image: np.ndarray, depth_image: np.ndarray,
                 camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = None) -> Dict:
    detector = OpenCVDetectors()
    if camera_matrix is not None and dist_coeffs is not None:
        detector.set_camera_calibration(camera_matrix, dist_coeffs)

    detection_methods = {
        'apriltag_36h11': detector.detect_apriltag_36h11,
        'aruco_4x4_50': detector.detect_aruco_4x4_50,
        'aruco_6x6_250': detector.detect_aruco_6x6_250,
        'qr_code': detector.detect_qr_code
    }

    if marker_type in detection_methods:
        try:
            return detection_methods[marker_type](rgb_image, depth_image)
        except Exception as e:
            return {
                'detected': False,
                'marker_type': marker_type,
                'error': f'Exception during detection: {str(e)}'
            }
    else:
        return {
            'detected': False,
            'marker_type': marker_type,
            'error': f'Marker type {marker_type} not supported by OpenCV detectors'
        }
