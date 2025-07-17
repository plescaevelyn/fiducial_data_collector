# Fiducial Data Collector - Systematic Data Collection

## General Overview

This project focuses **exclusively on data collection** for fiducial markers using the OAK-D Lite. The system guides the user step-by-step through all necessary measurements.

## Inspired by CopperTag Research

Based on the methodology of the CopperTag paper, but adapted for:
- Real-world measurements (no simulation)
- Stereo-depth camera (OAK-D Lite)
- Interactive user guidance
- Optimized test combinations for a single operator

## Optimized Test Combinations

### **Selected Markers (7 representative types)**
1. **ArUco 4x4_50** – Industrial standard
2. **ArUco 6x6_250** – Balanced precision/speed
3. **AprilTag 36h11** – Most robust AprilTag
4. **QR Code** – Commercial standard
5. **RuneTag** – Circular representative
6. **ChromaTag** – Color-based representative
7. **CopperTag** – Robust industrial representative

### **Test Conditions (Inspired by CopperTag)**

#### **Test Set 1: Distance (8 measurements × 7 markers = 56 tests)**
- 0.3m, 0.6m, 1.0m, 1.5m, 2.0m, 2.5m, 3.0m, 3.5m
- **Estimated time**: ~2 hours

#### **Test Set 2: X-Rotation (7 × 7 = 49 tests)**
- -60°, -40°, -20°, 0°, 20°, 40°, 60°
- **Estimated time**: ~1.5 hours

#### **Test Set 3: Y-Rotation (7 × 7 = 49 tests)**
- -60°, -40°, -20°, 0°, 20°, 40°, 60°
- **Estimated time**: ~1.5 hours

#### **Test Set 4: Z-Rotation (5 × 7 = 35 tests)**
- 0°, 45°, 90°, 180°, 270°
- **Estimated time**: ~1 hour

#### **Test Set 5: Occlusion (4 × 7 = 28 tests)**
- 5%, 10%, 15%, 20% (with physical objects)
- **Estimated time**: ~1 hour

#### **Test Set 6: Lighting (4 × 7 = 28 tests)**
- Bright, Normal, Dim, Shadow
- **Estimated time**: ~1 hour

**TOTAL: 245 tests in ~7 hours of collection**

## Project Structure

```
fiducial_data_collector/
├── README.md                    # This file
├── main_collector.py            # Main script with interactive guidance
├── config/
│   ├── test_configurations.py   # Test setup configurations
│   ├── marker_definitions.py    # Marker definitions
│   └── measurement_protocol.py  # Measurement protocol
├── detectors/
│   ├── opencv_detectors.py      # OpenCV-based detectors (ArUco, AprilTag, QR)
│   ├── external_detectors.py    # External detectors (RuneTag, ChromaTag, etc.)
│   └── detector_manager.py      # Detector orchestration logic
├── data_collection/
│   ├── oak_interface.py         # OAK-D Lite interface
│   ├── metrics_collector.py     # Metric collection logic
│   ├── user_guidance.py         # User instruction logic
│   └── data_saver.py            # Data persistence
├── utils/
│   ├── system_monitor.py        # CPU/RAM monitoring
│   ├── progress_tracker.py      # Track collection progress
│   └── validation_helpers.py    # Data validation utilities
├── markers/                     # Printable markers
│   ├── aruco/
│   ├── apriltag/
│   ├── qr/
│   └── custom/
└── datasets/                    # Collected datasets
    ├── raw_data/
    ├── processed/
    └── reports/
```

## Collection Flow

### **Step 1: Preparation**
```
Print markers from markers/
Prepare a ruler for distance tests
Prepare lighting sources for illumination tests
Prepare occlusion objects (e.g. paper, cardboard)
```

### **Step 2: Calibration**
```
Connect OAK-D Lite
Automatic camera calibration
Set up coordinate system
```

### **Step 3: Guided Collection**
```
The system gives precise instructions:
  "Place ArUco 4x4_50 marker at 0.3m distance"
  "Rotate the camera 20° on X-axis"
  "Apply shadow on half of the marker"

Automatically collects all metrics
Saves data in real-time
Displays progress (Test 15/280)
```

## Interactive Guidance

### **Example Interaction**
```
FIDUCIAL DATA COLLECTOR
Progress: 15/280 tests (5.4%)
Estimated time remaining: 7h 23min

CURRENT TEST: Distance - ArUco 4x4_50
Instructions:
  1. Print ArUco 4x4_50 marker (5cm x 5cm)
  2. Attach it to a flat surface
  3. Position it at EXACTLY 0.6m from the camera
  4. Ensure it is perpendicular to the camera
  5. Press ENTER when ready

Camera status: Marker found
Measured distance: 0.58m (±2cm - OK)
Collecting... 10s

Test complete!
Results:
  - Detection rate: 98.5%
  - Processing time: 12.3ms
  - CPU: 45%, RAM: 1.2GB
  - Corners detected: 4/4

Next test: ArUco 4x4_50 at 1.0m
```

## Collected Metrics

### **For Each Test (10 metrics)**
1. **CPU usage** - during detection
2. **Memory usage** - peak MB used
3. **Processing time** - per frame
4. **Detection rate** - % of frames with valid detection
5. **Measured distance** - vs ground truth
6. **Corner precision** - pixel error
7. **ID stability** - consistent identification
8. **Motion robustness** - detection under motion
9. **Depth quality** - depth data validity
10. **Overall score** - aggregate rating

## Advantages of This Approach

### **Compared to CopperTag**
- Real measurements instead of simulation
- Depth data enables accurate 3D positioning
- Step-by-step interactive guidance
- Optimized for one-person operation (~8h vs weeks)

### **Compared to Other Studies**
- Diverse marker set (7 representative types)
- Realistic testing conditions (lighting, occlusion)
- Full metric suite (10 types)
- Structured datasets for post-analysis

## Expected Outcomes

### **Final Dataset**
- **245 complete tests**
- **~45GB of data** (RGB + Depth + Metadata)
- **2450 individual metrics**
- **Auto-generated statistical report**

### **Use Cases**
- Academic research (dataset for publications)
- Industrial development (choose optimal markers)
- Benchmarking (objective algorithm comparisons)
- Optimization (identify weaknesses and improve)

## Next Steps

1. Implement main interactive script
2. Integrate all marker detectors
3. Validate tests with OAK-D Lite
4. Collect the dataset in ~8 hours
5. Generate and review automatic report