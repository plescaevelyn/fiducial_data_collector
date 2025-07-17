# Synthetic Fiducial Marker Dataset Generator

This module provides a **complete synthetic data generator** for producing realistic test sets of fiducial markers under controlled conditions.  
It provides an **offline, deterministic generator** that outputs:

- **RGB images** (PNG, 8-bit)
- **Depth maps** (uint16 PNG, mm units)
- **Disparity maps** (uint16 PNG, 1/16-px OAK-D style)
- **Segmentation masks** (uint8 PNG)
- A global **annotations.csv** with metadata for every sample

## Supported Marker Types

The generator creates representative versions of **seven** fiducial families:

### Standard Families
1. **QR Code** (via the `qrcode` Python package)  
2. **ArUco 4×4 (50)**  
3. **ArUco 6×6 (250)**  
4. **AprilTag 36h11** *(requires opencv-contrib)*

### Representative “research family” markers
5. **RuneTag** — circular marker with ring + blobs  
6. **ChromaTag** — color-based marker with 2×2 color blocks  
7. **CopperTag** — industrial-style 7×7 grid with robust border  

These last three are *representative, not exact,* but match category characteristics for testing.

## Test Sets Produced

For every marker family, the generator creates four complete test groups:

### **1. Distance Test**
Marker shown at:

```
0.3 m, 0.6 m, 1.0 m, 1.5 m, 2.0 m,
2.5 m, 3.0 m, 3.5 m
```

### **2. Z-Rotation Test (In-plane)**
```
0°, 45°, 90°, 180°, 270°
```

### **3. Occlusion Test**
Vertical occlusions covering:

```
5%, 10%, 15%, 20%
```

### **4. Lighting Test**
Four lighting conditions:

- `bright`
- `normal`
- `dim`
- `shadow` (left-to-right darkening gradient)

Each sample includes:  
**RGB + depth_mm + disparity + segmentation + metadata.**

## Synthetic Scene

The scene approximates an OAK-D Lite RGB camera environment:

- Resolution: **1280×720**
- Horizontal FOV: **71.86°**
- Stereo baseline: **7.5 cm**
- Marker size: **10 cm**
- Wall + floor structure for realism (non-flat background)
- Marker placed on the wall at mid-height
- Depth:
  - Wall depth = distance
  - Floor is slightly further (1.2× distance)

## Output Structure

Generated files appear in:

```
fiducial_data_collector/datasets/synthetic_markers/
```

Example:

```
synthetic_markers/
    qr_distance/
        rgb_*.png
        depth_mm_*.png
        disp_*.png
        seg_*.png
    aruco_4x4_50_z_rotation/
    apriltag_36h11_occlusion/
    chromatag_lighting/
    coppertag_distance/
    runetag_distance/
    annotations.csv
```

**Modalities**

| File | Format | Meaning |
|------|--------|---------|
| `rgb_*.png` | uint8 PNG | RGB render |
| `depth_mm_*.png` | uint16 PNG | Depth in millimeters |
| `disp_*.png` | uint16 PNG | Disparity ×16 (OAK-D style) |
| `seg_*.png` | uint8 PNG | 0=bg, 1=wall, 2=floor, 3=marker |

The CSV contains for each sample:

- marker type & id  
- distance  
- rotation  
- occlusion percentage  
- lighting condition  
- intrinsics, baseline, image size  
- relative paths to all outputs  

## Running the Generator

From repo root:

```bash
cd fiducial_data_collector
python3 synthetic_generation/generate_synthetic_markers_dataset.py
```

The dataset is generated in a few seconds.

## Dependencies

Install required packages:

```bash
pip install qrcode pillow numpy opencv-contrib-python
```

> `opencv-contrib-python` is **required** for ArUco + AprilTag generation.

If your OpenCV version does not include AprilTag support, disable the corresponding line in `marker_families`.

## Using the Synthetic Dataset

You can use any of your existing OpenCV-based detectors on these images:

```python
import cv2
import pandas as pd

df = pd.read_csv("datasets/synthetic_markers/annotations.csv")

for _, row in df.iterrows():
    rgb = cv2.imread(row['rgb_path'])
    depth = cv2.imread(row['depth_path'], cv2.IMREAD_UNCHANGED)
    
    detections = detect_qr(rgb)  # your existing detector
    ...
```

This dataset integrates directly with your metrics generator and evaluation pipeline.

## Summary

This generator provides:

- Fully synthetic dataset  
- Covers **7 marker families**  
- Full modality set: RGB / depth / disparity / segmentation  
- Matching your **real-camera test structure**  
- Deterministic & reproducible  
- Ideal for training, benchmarking, and scientific evaluation
