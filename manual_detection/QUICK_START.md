# Quick Start Guide - Fiducial Data Collector

## Overview

This system collects data for **8 types of markers** across **280 optimized tests**, inspired by the **CopperTag** methodology but adapted for real-world measurements with **OAK-D Lite**.

## Quick Installation

```bash
# 1. Install dependencies
pip install depthai opencv-python numpy psutil

# 2. Check OAK-D Lite detection
python -c "import depthai as dai; print('Detected:', len(dai.Device.getAllAvailableDevices()))"

# 3. Run the collector
cd fiducial_data_collector
python main_collector.py
```

## What You'll Collect

### **Tested Marker Types (8)**
1. **ArUco 4x4_50** - Industrial standard
2. **ArUco 6x6_250** - Improved precision
3. **AprilTag 36h11** - Most robust
4. **QR Code** - Commercial standard
5. **RuneTag** - Circular, occlusion-resistant
6. **ChromaTag** - Color-based
7. **CopperTag** - Robust industrial
8. **Custom Hybrid** - Custom design

### **Test Scenarios (280 total)**
- **64 distance tests** - 0.3m → 3.5m (8 distances × 8 markers)
- **56 X-rotation tests** - -60° → +60° (7 angles × 8 markers)
- **56 Y-rotation tests** - -60° → +60° (7 angles × 8 markers)
- **40 Z-rotation tests** - 0°, 45°, 90°, 180°, 270° (5 × 8 markers)
- **32 occlusion tests** - 5%, 10%, 15%, 20% (4 × 8 markers)
- **32 lighting tests** - bright, normal, dim, shadow (4 × 8 markers)

**Estimated total duration: ~8 hours**

## Preparation (5 minutes)

### **Required Materials**
- OAK-D Lite connected to laptop
- Printed markers (5cm × 5cm each)
- Ruler for measuring distances
- Adjustable lighting source
- Occlusion props (paper, cardboard)
- Protractor for angle measurements
- Stable camera mount

### **Printing the Markers**
```bash
# Markers are located in the markers/ directory
ls markers/
# aruco/     apriltag/     qr/     custom/

# Print each marker at EXACTLY 5cm × 5cm
# Use matte white paper (not glossy)
```

## Running the Data Collector

### **Main Command**
```bash
python main_collector.py
```

### **Example Interaction**
```
FIDUCIAL DATA COLLECTOR
==================================================
Total tests: 280
Estimated time: 8.0 hours
Session: 20250115_143022

Starting data collection

System Initialization...
   Pipeline created
   OAK-D Lite detected: 18443010D1389ACD00

============================================================
TEST 1/280
============================================================
Testing aruco_4x4_50 at 0.3m
Type: distance
Marker: aruco_4x4_50
Parameter: 0.3 m
Estimated duration: 90s

SETUP INSTRUCTIONS:
   Place the marker at 0.3m from the camera

MARKER: ArUco 4x4_50
File: markers/aruco/aruco_4x4_50_id_0.pdf
Print size: 5cm x 5cm

DISTANCE TEST: 0.3m
Setup Instructions:
   1. Place the marker on a stable surface
   2. Measure EXACTLY 0.3m between the camera and marker
   3. Ensure the marker is perpendicular to the camera
   4. The marker should be centered in the image
   5. Make sure the entire marker is visible

Setup complete? (y/n/s for skip): y

Collecting data...
Detecting marker...
   Progress: 30% - Detections: 28/90
   Progress: 60% - Detections: 55/180
   Progress: 100% - Detections: 89/300

Test complete!
Frames processed: 300
Detections: 89
Detection rate: 29.7%

GENERAL PROGRESS:
Completed: 1/280 (0.4%)
Elapsed time: 0.0h
Estimated remaining time: 7.8h
```

## Test Types

### **1. Distance Tests**
- **Instruction**: Place the marker at exact distances
- **Distances**: 0.3m, 0.6m, 1.0m, 1.5m, 2.0m, 2.5m, 3.0m, 3.5m
- **Goal**: Measure performance vs distance

### **2. X-Rotation Tests (Pitch)**
- **Instruction**: Tilt the camera up/down
- **Angles**: -60°, -40°, -20°, 0°, 20°, 40°, 60°
- **Goal**: Test vertical angle robustness

### **3. Y-Rotation Tests (Yaw)**
- **Instruction**: Rotate the camera left/right
- **Angles**: -60°, -40°, -20°, 0°, 20°, 40°, 60°
- **Goal**: Test horizontal angle robustness

### **4. Z-Rotation Tests (Roll)**
- **Instruction**: Rotate the marker in its own plane
- **Angles**: 0°, 45°, 90°, 180°, 270°
- **Goal**: Test robustness to in-plane rotation

### **5. Occlusion Tests**
- **Instruction**: Cover part of the marker
- **Levels**: 5%, 10%, 15%, 20%
- **Goal**: Test robustness to partial obstruction

### **6. Lighting Tests**
- **Instruction**: Change light conditions
- **Conditions**: Bright, Normal, Dim, Shadow
- **Goal**: Test performance under variable lighting

## Collected Data

### **For Each Test**
```
datasets/raw_data/test_001_distance_aruco_4x4_50/
├── rgb.jpg              # RGB image
├── depth_raw.npy        # Raw depth data
├── depth_vis.png        # Visualized depth
├── edge.png             # Edge detection
└── test_data.json       # All collected metrics
```

### **Collected Metrics (10 categories)**
1. **CPU usage** - during detection
2. **Memory usage** - peak MB used
3. **Processing time** - ms per frame
4. **Detection rate** - % of successful frames
5. **Measured distance** - vs actual
6. **Corner precision** - pixel error
7. **ID stability** - consistency across frames
8. **Motion robustness** - detection during movement
9. **Depth quality** - validity of depth data
10. **Overall score** - aggregate metric

## Final Results

### **Full Dataset**
- **280 complete tests**
- **~50GB of data** (RGB + Depth + Metadata)
- **2800 individual metrics**
- **Automatic report** with stats

### **Final Report**
```json
{
  "session_id": "20250115_143022",
  "completion_time": "2025-01-15T22:30:22",
  "total_tests": 280,
  "completed_tests": 280,
  "success_rate": 100.0,
  "total_duration_hours": 7.8
}
```

## Advantages vs Other Studies

### **Compared to CopperTag**
- Real-world measurements (vs Unity simulation)
- Depth data for 3D positioning
- Step-by-step guided setup
- Designed for single-operator collection (~8h vs weeks)

### **Compared to Other Benchmarks**
- 8 representative marker types
- Realistic conditions (lighting, occlusion)
- 10 complete metrics (not just detection rate)
- Structured data for analysis

## Troubleshooting

### **OAK-D Lite not detected**
```bash
lsusb | grep "03e7"
pip uninstall depthai && pip install depthai --upgrade
```

### **Low detection rate**
- Check marker print (must be exactly 5cm)
- Ensure marker is flat and centered
- Avoid reflections and direct glare
- Adjust distance and viewing angle

### **Slow data collection**
- Close other applications
- Check disk space
- Lower camera resolution if needed

## Next Steps

After data collection:
1. Statistical analysis of the dataset
2. Identify best-performing markers
3. Develop optimized detection algorithms
4. Publish scientific results
