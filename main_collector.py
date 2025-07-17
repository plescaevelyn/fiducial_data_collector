# Main script for guided data collection
# Inspired by the CopperTag methodology, optimized for real-world measurements using the OAK-D Lite

import cv2
import depthai as dai
import numpy as np
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import configurations
from config.test_configurations import (
    generate_all_test_configurations, 
    get_test_summary,
    TestConfiguration,
    TestType,
    MarkerType
)

class FiducialDataCollector:
    """
    Main interactive data collector class.
    
    FEATURES:
    - Interactive guidance for each test
    - Integration with scan_sticla.py
    - Automatic metric collection
    - Real-time progress tracking
    - Setup validation
    """
    
    def __init__(self):
        self.output_dir = Path("datasets/raw_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_configs = generate_all_test_configurations()
        self.test_summary = get_test_summary()
        
        self.current_test_index = 0
        self.completed_tests = 0
        self.start_time = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.pipeline = None
        self.device = None
        
        print("FIDUCIAL DATA COLLECTOR")
        print("=" * 50)
        print(f"Total tests: {self.test_summary['total_tests']}")
        print(f"Estimated time: {self.test_summary['estimated_hours']:.1f} hours")
        print(f"Session ID: {self.session_id}")
    
    def create_oak_pipeline(self) -> dai.Pipeline:
        """
        Creates the OAK-D Lite pipeline (based on scan_sticla.py).
        Optimized for fiducial marker detection.
        """
        pipeline = dai.Pipeline()
        
        mono_left = pipeline.createMonoCamera()
        mono_right = pipeline.createMonoCamera()
        
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        stereo = pipeline.createStereoDepth()
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        
        manip = pipeline.createImageManip()
        manip.initialConfig.setResize(640, 480)
        cam_rgb.video.link(manip.inputImage)
        
        edge = pipeline.createEdgeDetector()
        manip.out.link(edge.inputImage)
        
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.video.link(xout_rgb.input)
        
        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        
        xout_edge = pipeline.createXLinkOut()
        xout_edge.setStreamName("edge")
        edge.outputImage.link(xout_edge.input)
        
        return pipeline
    
    def run_data_collection(self):
        """
        Runs the full data collection process with step-by-step guidance.
        """
        print("\nStarting data collection...")
        
        if not self._initialize_system():
            return
        
        self._show_initial_guidance()
        
        try:
            with dai.Device(self.pipeline) as device:
                self.device = device
                self._setup_device_queues()
                
                self.start_time = time.time()
                
                for i, test_config in enumerate(self.test_configs):
                    self.current_test_index = i
                    
                    if not self._run_single_test(test_config):
                        break
                    
                    self.completed_tests += 1
                    self._show_progress()
                
                self._finalize_collection()
                
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            print("Data collection finished")
    
    def _initialize_system(self) -> bool:
        """Initializes system and checks device connection."""
        print("\nInitializing system...")
        
        try:
            self.pipeline = self.create_oak_pipeline()
            print("   Pipeline created successfully")
            
            devices = dai.Device.getAllAvailableDevices()
            if not devices:
                print("   OAK-D Lite not detected!")
                return False
            
            print(f"   Device detected: {devices[0].getMxId()}")
            return True
            
        except Exception as e:
            print(f"   Initialization error: {e}")
            return False
    
    def _show_initial_guidance(self):
        """Displays preparation checklist before starting."""
        print("\n" + "=" * 60)
        print("PRE-COLLECTION CHECKLIST")
        print("=" * 60)
        print("Before starting, make sure you have:")
        print("1. Printed markers (5cm x 5cm each)")
        print("2. Ruler for measuring distances")
        print("3. Light sources for lighting tests")
        print("4. Occlusion objects (paper, cardboard)")
        print("5. Stable camera support")
        print("6. Protractor for measuring angles")
        
        input("\nPress ENTER when ready to begin...")
    
    def _setup_device_queues(self):
        """Configures device queues."""
        self.rgb_queue = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self.depth_queue = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
        self.edge_queue = self.device.getOutputQueue("edge", maxSize=4, blocking=False)
    
    def _run_single_test(self, config: TestConfiguration) -> bool:
        """
        Runs a single guided test.
        
        Returns:
            True if test completed, False if user stopped
        """
        print("\n" + "=" * 60)
        print(f"TEST {self.current_test_index + 1}/{len(self.test_configs)}")
        print("=" * 60)
        print(f"Description: {config.description}")
        print(f"Type: {config.test_type.value}")
        print(f"Marker: {config.marker_type.value}")
        print(f"Parameter: {config.parameter_value} {config.parameter_unit}")
        print(f"Estimated duration: {config.estimated_duration_seconds}s")
        
        print("\nINSTRUCTIONS:")
        print(f"   {config.special_instructions}")
        
        response = input("\nPress ENTER to start the test (or 'q' to quit): ")
        if response.lower() == 'q':
            return False
        
        return self._execute_test_measurement(config)
    
    def _execute_test_measurement(self, config: TestConfiguration) -> bool:
        """Performs the actual measurement for a test."""
        print("Collecting...")

        test_data = {
            'config': config.__dict__,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'measurements': []
        }
        
        collection_duration = 10
        start_time = time.time()
        frame_count = 0
        
        print("Detecting marker...")

        while (time.time() - start_time) < collection_duration:
            rgb_data = self.rgb_queue.get()
            depth_data = self.depth_queue.get()
            edge_data = self.edge_queue.get()
            
            if rgb_data and depth_data and edge_data:
                rgb_frame = rgb_data.getCvFrame()
                depth_frame = depth_data.getFrame()
                edge_frame = edge_data.getCvFrame()
                
                cv2.imshow("RGB Frame", rgb_frame)
                cv2.waitKey(1)
            
                detection_result = self._detect_marker(
                    config.marker_type, rgb_frame, depth_frame
                )

                if detection_result['detected']:
                    measurement = {
                        'frame_id': frame_count,
                        'timestamp': time.time(),
                        'detection_result': detection_result,
                        'system_metrics': self._collect_system_metrics()
                    }
                    test_data['measurements'].append(measurement)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    progress = (elapsed / collection_duration) * 100
                    detections = len(test_data['measurements'])
                    print(f"Progress: {progress:.0f}% - Detections: {detections}/{frame_count}")
        
        self._save_test_data(config, test_data, rgb_frame, depth_frame, edge_frame)
        
        total_detections = len(test_data['measurements'])
        detection_rate = (total_detections / frame_count) * 100 if frame_count > 0 else 0
        
        print("Test completed.")
        print(f"Frames processed: {frame_count}")
        print(f"Successful detections: {total_detections}")
        print(f"Detection rate: {detection_rate:.1f}%")
        
        return True
    
    def _detect_marker(self, marker_type: MarkerType, rgb_frame: np.ndarray, 
                      depth_frame: np.ndarray) -> Dict:
        """
        Detects the specified marker using DetectorManager.
        """
        if not hasattr(self, 'detector_manager'):
            from detectors.detector_manager import DetectorManager
            self.detector_manager = DetectorManager()
            
            if hasattr(self, 'device') and self.device:
                self.detector_manager.calibrate_camera(self.device)
        
        result = self.detector_manager.detect_marker(
            marker_type.value, rgb_frame, depth_frame
        )
        
        return result
    
    def _collect_system_metrics(self) -> Dict:
        """Collects system-level metrics (CPU, memory)."""
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_mb': psutil.virtual_memory().used / (1024 * 1024),
            'timestamp': time.time()
        }
    
    def _save_test_data(self, config: TestConfiguration, test_data: Dict,
                       rgb_frame: np.ndarray, depth_frame: np.ndarray, 
                       edge_frame: np.ndarray):
        """Saves all test-related data to disk."""
        test_dir = self.output_dir / f"test_{self.current_test_index + 1:03d}_{config.test_type.value}_{config.marker_type.value}"
        test_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(test_dir / "rgb.jpg"), rgb_frame)
        cv2.imwrite(str(test_dir / "edge.png"), edge_frame)
        
        np.save(str(test_dir / "depth_raw.npy"), depth_frame)
        
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.convertScaleAbs(depth_vis)
        cv2.imwrite(str(test_dir / "depth_vis.png"), depth_vis)
        
        with open(test_dir / "test_data.json", "w") as f:
            json.dump(test_data, f, indent=2, default=str)
    
    def _show_progress(self):
        """Displays overall progress."""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            progress_percent = (self.completed_tests / len(self.test_configs)) * 100
            
            if self.completed_tests > 0:
                avg_time_per_test = elapsed_time / self.completed_tests
                remaining_tests = len(self.test_configs) - self.completed_tests
                estimated_remaining = remaining_tests * avg_time_per_test
                
                print("\nOVERALL PROGRESS:")
                print(f"Completed: {self.completed_tests}/{len(self.test_configs)} ({progress_percent:.1f}%)")
                print(f"Elapsed time: {elapsed_time/3600:.1f} hours")
                print(f"Estimated time remaining: {estimated_remaining/3600:.1f} hours")
    
    def _finalize_collection(self):
        """Finalizes the collection and generates a summary report."""
        print("\nCOLLECTION COMPLETE")
        
        report = {
            'session_id': self.session_id,
            'completion_time': datetime.now().isoformat(),
            'total_tests': len(self.test_configs),
            'completed_tests': self.completed_tests,
            'success_rate': (self.completed_tests / len(self.test_configs)) * 100,
            'total_duration_hours': (time.time() - self.start_time) / 3600 if self.start_time else 0
        }
        
        with open(self.output_dir / f"collection_report_{self.session_id}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved: collection_report_{self.session_id}.json")
        print(f"Data stored in: {self.output_dir}")

def main():
    collector = FiducialDataCollector()
    collector.run_data_collection()

if __name__ == "__main__":
    main()
