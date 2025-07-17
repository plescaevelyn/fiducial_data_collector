# Fiducial Marker Dataset Generator & Collector  
**Synthetic + Manual Data Acquisition Framework**

This project provides a complete system for generating and collecting data for
fiducial marker detection algorithms.  
It supports two independent workflows:

1. **Synthetic dataset generation** (camera-independent, fully controlled)
2. **Manual real-world data collection** using **OAK-D Lite**

You can choose **either method** depending on your use case — or combine both for
maximum robustness.


# 1. Synthetic Dataset Generation (Recommended)

The **synthetic pipeline** creates perfectly controlled ground-truth data using
*BlenderProc*, simulating:

- RGB images  
- Depth maps (mm)  
- Disparity maps  
- Camera intrinsics  
- Lighting variations  
- Occlusions  
- Pose + 3D ground truth  
- All marker types:
  - QR Code  
  - ArUco 4×4 50  
  - ArUco 6×6 250  
  - AprilTag 25h9  
  - AprilTag 36h11  
  - RuneTag (synthetic)  
  - ChromaTag (synthetic)  
  - CopperTag (synthetic)

> **Best for:** algorithm development, benchmarking, training ML models,
> testing corner cases, replicating CopperTag-style experiments.

## Synthetic Requirements

- BlenderProc 3.x  
- Blender 4.2 LTS  
- Python 3.10  
- GPU (NVIDIA recommended)  
- `OpenEXR`, `numpy`, `opencv-python`  

## Running Synthetic Generation

```bash
cd synthetic_generation
python3 generate_synthetic_markers_dataset.py
```

Outputs are stored in:

```
datasets/synthetic_markers/<marker>/<distance>/<angle>/<lighting>/...
```

Each sample includes:

- rgb.png
- depth_mm.png
- disp.png
- camera_intrinsics.json
- metadata.json (ground-truth pose, angle, occlusion, lighting)
- segmentation masks

# 2. Manual Data Collection (Real OAK-D Lite)

The **manual workflow** performs guided, systematic data acquisition using the
actual **OAK-D Lite** camera.  
It recreates real CopperTag-style measurements:

- Multiple distances (0.3–2.0 m)
- X/Y/Z rotations
- Lighting conditions
- Occlusion levels
- All marker families (QR, ArUco, AprilTag, RuneTag, ChromaTag, CopperTag)

The user is guided interactively step-by-step.

> **Best for:** validating real hardware behavior, building real datasets,
> comparing physical vs. synthetic performance.

## Manual Requirements

### **Hardware**
- Camera: Luxonis **OAK-D Lite**
- **Measuring tape** (for accurate distance calibration)
- **Goniometer** (for angle measurement)
- Optional: adjustable lights, occlusion objects

### **Software**
- Ubuntu **22.04**
- ROS 2 **Humble**
- Python **3.10.12**
- DepthAI **2.30.0.0**
- OpenCV **4.9.0.80**
- NumPy **1.26.4**

---

## Running Manual Collection

```bash
cd manual_detection
python3 run_collector.py
```

You will be guided through:

- Positioning the marker  
- Aligning distances  
- Rotating at precise angles  
- Adjusting illumination  
- Introducing occlusion  
- Capturing RGB + depth + edge maps  
- Recording all metrics + detection statistics  

Data is saved in:

```
datasets/raw_data/test_XXX_<type>_<marker>/
```

Each folder contains:

- rgb.jpg
- depth_raw.npy
- depth_vis.png
- edge.png
- test_data.json

# When Should You Use Synthetic vs Manual?

| Goal | Synthetic | Manual |
|------|-----------|--------|
| Fast generation of large datasets | ✅ | ❌ |
| Perfect control over ground truth | ✅ | ❌ |
| Replicate CopperTag metrics | ✅ | ⚠️ possible but slower |
| Validate algorithm on *real* sensor noise | ⚠️ optional | ✅ |
| Compare ideal vs. real camera | combine both |
| Train ML models | ✅ | optional |
| Evaluate physical distortions | ❌ | ✅ |

**Recommended workflow:**  
1. Develop & tune using **synthetic** data  
2. Validate on **OAK-D Lite**  
3. Compare results for sensor-specific biases

# Project Structure

```
project_root/
│
├── synthetic_generation/       # Synthetic dataset generator (BlenderProc)
│   └── generate_synthetic_markers_dataset.py
│
├── manual_detection/           # Real camera data collection
│   └── run_collector.py
│
├── detectors/                  # Unified OpenCV + external detector stack
├── config/                     # Test plan generation (distance / angle / lighting)
├── markers/                    # Pre-rendered marker templates
│
├── datasets/
│   ├── synthetic_markers/      # Output of synthetic generator
│   └── raw_data/               # Output of manual collector
│
└── README.md
```

# Summary

This repository gives you:

- Synthetic generation (fast, perfect ground truth)  
- Manual collection (real-world distortions)  
- Multi-marker support (QR, ArUco, AprilTag, RuneTag, ChromaTag, CopperTag)  
- A unified evaluation pipeline for systematic fiducial marker analysis

Both systems complement each other, enabling a complete,
scientifically repeatable evaluation environment.
