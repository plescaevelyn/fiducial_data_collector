#!/usr/bin/env python3
# Debug script for AprilTag 36h11
# Systematic steps to identify detection issues

import cv2
import numpy as np
import depthai as dai
from pathlib import Path
import time

def debug_apriltag_detection():
    """
    Systematic debugging steps for AprilTag 36h11.
    Quickly identifies root cause of detection issues.
    """

    print("DEBUG APRILTAG 36H11")
    print("=" * 50)

    # STEP 1: Check OpenCV and dictionary
    print("\nSTEP 1: Check OpenCV and dictionary")
    try:
        cv_version = cv2.__version__
        print(f"   OpenCV version: {cv_version}")

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        print(f"   AprilTag 36h11 dictionary loaded: {dictionary.bytesList.shape[0]} markers")

        params = cv2.aruco.DetectorParameters()
        print(f"   Detector parameters created")

    except Exception as e:
        print(f"   OpenCV ERROR: {e}")
        return False

    # STEP 2: OAK-D Lite check
    print("\nSTEP 2: Check OAK-D Lite connection")
    try:
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("   OAK-D Lite not detected!")
            return False

        print(f"   OAK-D Lite detected: {devices[0].getMxId()}")

        pipeline = dai.Pipeline()
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)

        xout = pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        cam_rgb.preview.link(xout.input)

        print("   Pipeline created")

    except Exception as e:
        print(f"   OAK-D ERROR: {e}")
        return False

    # STEP 3: Image capture and analysis
    print("\nSTEP 3: Capture and analyze image")
    input("   Position the AprilTag in front of the camera and press ENTER...")

    try:
        with dai.Device(pipeline) as device:
            q = device.getOutputQueue("rgb")
            rgb_frame = q.get().getCvFrame()
            print(f"   Image captured: {rgb_frame.shape}")

            cv2.imwrite("debug_apriltag_capture.jpg", rgb_frame)
            print("   Saved image: debug_apriltag_capture.jpg")

            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            print(f"   Mean brightness: {mean_brightness:.1f}")
            print(f"   Std deviation: {std_brightness:.1f}")

            if mean_brightness < 50:
                print("   Image too dark!")
            elif mean_brightness > 200:
                print("   Image too bright!")

            if std_brightness < 20:
                print("   Low contrast warning!")

    except Exception as e:
        print(f"   CAPTURE ERROR: {e}")
        return False

    # STEP 4: Try detection with different parameters
    print("\nSTEP 4: Try detection with different parameters")

    base_params = cv2.aruco.DetectorParameters()
    relaxed_params = cv2.aruco.DetectorParameters()
    relaxed_params.adaptiveThreshWinSizeMin = 3
    relaxed_params.adaptiveThreshWinSizeMax = 50
    relaxed_params.minMarkerPerimeterRate = 0.01
    relaxed_params.maxMarkerPerimeterRate = 8.0
    relaxed_params.polygonalApproxAccuracyRate = 0.05
    relaxed_params.minCornerDistanceRate = 0.01
    relaxed_params.minDistanceToBorder = 1

    strict_params = cv2.aruco.DetectorParameters()
    strict_params.adaptiveThreshWinSizeMin = 5
    strict_params.adaptiveThreshWinSizeMax = 15
    strict_params.minMarkerPerimeterRate = 0.05
    strict_params.maxMarkerPerimeterRate = 2.0

    param_sets = [
        ("Base Parameters", base_params),
        ("Relaxed Parameters", relaxed_params),
        ("Strict Parameters", strict_params)
    ]

    for name, params in param_sets:
        print(f"\n   Testing with {name}:")
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        print(f"      Detected markers: {len(ids) if ids is not None else 0}")
        print(f"      Rejected candidates: {len(rejected)}")

        if ids is not None and len(ids) > 0:
            print(f"      SUCCESS! IDs: {[int(id[0]) for id in ids]}")
            debug_image = rgb_frame.copy()
            cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
            cv2.imwrite(f"debug_apriltag_detected_{name.lower().replace(' ', '_')}.jpg", debug_image)
            print(f"      Detection result saved")
            return True
        else:
            print(f"      No detections")

    # STEP 5: Rejected candidate analysis
    print("\nSTEP 5: Analyze rejected candidates")
    if len(rejected) > 0:
        print(f"   {len(rejected)} rejected candidates")
        debug_image = rgb_frame.copy()
        for i, candidate in enumerate(rejected):
            if len(candidate) >= 4:
                pts = candidate.astype(np.int32)
                cv2.polylines(debug_image, [pts], True, (0, 0, 255), 2)
                center = np.mean(pts, axis=0).astype(int)
                cv2.putText(debug_image, f"R{i}", tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite("debug_apriltag_rejected.jpg", debug_image)
        print("   Rejected candidates image saved")

        candidate = rejected[0]
        area = cv2.contourArea(candidate)
        perimeter = cv2.arcLength(candidate, True)
        print(f"   Area: {area:.1f} px², Perimeter: {perimeter:.1f} px")

        if area < 100:
            print("   Possibly too small")
        elif area > 50000:
            print("   Possibly too large")
    else:
        print("   No rejected candidates. Possible issues:")
        print("      - No marker in image")
        print("      - Low contrast")
        print("      - Damaged or distorted marker")

    # STEP 6: Remediation suggestions
    print("\nSTEP 6: Suggestions")
    if mean_brightness < 50:
        print("   Increase lighting")
    elif mean_brightness > 200:
        print("   Decrease lighting")

    if std_brightness < 20:
        print("   Increase contrast")

    print("   Use 5–10 cm sized markers")
    print("   Keep marker flat")
    print("   Center the marker in frame")
    print("   Try distances between 0.5–2 meters")
    return False

def quick_test_apriltag():
    """Quick test for AprilTag detection with optimized parameters."""
    print("\nQUICK APRILTAG TEST")
    try:
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)

        xout = pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        cam_rgb.preview.link(xout.input)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 30
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 6.0
        params.polygonalApproxAccuracyRate = 0.04
        params.minCornerDistanceRate = 0.02
        params.minDistanceToBorder = 2

        with dai.Device(pipeline) as device:
            q = device.getOutputQueue("rgb")
            input("Position AprilTag and press ENTER...")

            rgb_frame = q.get().getCvFrame()
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

            start_time = time.perf_counter()
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
            detection_time = (time.perf_counter() - start_time) * 1000

            print(f"Detection time: {detection_time:.1f}ms")
            print(f"Detected markers: {len(ids) if ids is not None else 0}")
            print(f"Rejected candidates: {len(rejected)}")

            if ids is not None and len(ids) > 0:
                print(f"SUCCESS! IDs: {[int(id[0]) for id in ids]}")
                debug_image = rgb_frame.copy()
                cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
                cv2.imwrite("quick_test_apriltag_success.jpg", debug_image)
                print("Saved: quick_test_apriltag_success.jpg")
                return True
            else:
                print("No detections")
                cv2.imwrite("quick_test_apriltag_failed.jpg", rgb_frame)
                return False
    except Exception as e:
        print(f"Quick test ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Select debug option:")
    print("1. Full debug (recommended)")
    print("2. Quick test")

    choice = input("Option (1/2): ").strip()
    if choice == "2":
        quick_test_apriltag()
    else:
        debug_apriltag_detection()
