#!/usr/bin/env python3
import json
import sys
import csv
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

# -------------------------------------------------------------------
# Repo root detection & import path
# -------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
# REPO_ROOT = .../fiducial_data_collector
REPO_ROOT = THIS_FILE.parents[1]
print(f"[INFO] Repo root: {REPO_ROOT}")

# manual_detection/ is where detectors/ lives
MANUAL_ROOT = REPO_ROOT / "manual_detection"

# Make sure we can import detectors.detector_manager
for p in (str(REPO_ROOT), str(MANUAL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detectors.detector_manager import DetectorManager  # type: ignore

# -------------------------------------------------------------------
# Marker type inference from filename
# -------------------------------------------------------------------

def infer_marker_type_from_name(path: Path) -> Optional[str]:
    """
    Infer marker_type for DetectorManager from the RGB filename.
    """
    name = path.name.lower()

    if "aruco_4x4_50" in name:
        return "aruco_4x4_50"
    if "aruco_6x6_250" in name:
        return "aruco_6x6_250"
    if "apriltag_36h11" in name:
        return "apriltag_36h11"

    if "qr_" in name or name.startswith("rgb_qr_"):
        return "qr_code"

    if "runetag" in name:
        return "runetag"
    if "chromatag" in name:
        return "chromatag"
    if "coppertag" in name:
        return "coppertag"

    return None

# -------------------------------------------------------------------
# Locate synthetic_markers dir robustly
# -------------------------------------------------------------------

def find_synthetic_root() -> Optional[Path]:
    """
    Try several common locations, then fall back to a recursive search
    for a folder literally named 'synthetic_markers'.
    """
    candidates = [
        THIS_FILE.parent / "synthetic_markers",
        REPO_ROOT / "synthetic_markers",
        REPO_ROOT / "synthetic_generation" / "datasets" / "synthetic_markers",
    ]

    for cand in candidates:
        if cand.is_dir():
            print(f"[INFO] Found synthetic_markers at: {cand}")
            return cand.resolve()

    # Fallback: search anywhere under REPO_ROOT
    print("[INFO] Falling back to recursive search for 'synthetic_markers'...")
    for path in REPO_ROOT.rglob("*"):
        if path.name == "synthetic_markers" and path.is_dir():
            print(f"[INFO] Found synthetic_markers via search at: {path}")
            return path.resolve()

    return None

# -------------------------------------------------------------------
# Load annotations.csv (metadata from generator)
# -------------------------------------------------------------------

NUMERIC_FLOAT_KEYS = {
    "distance_m",
    "x_deg",
    "y_deg",
    "occlusion_pct",
    "qr_size_m",
    "fx",
    "fy",
    "cx",
    "cy",
    "baseline_m",
}
NUMERIC_INT_KEYS = {
    "img_width",
    "img_height",
    "marker_id",
}

def _parse_numeric_fields(row: Dict[str, str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            parsed[k] = None
            continue
        v = v.strip()
        if v == "":
            parsed[k] = None
            continue

        if k in NUMERIC_FLOAT_KEYS:
            try:
                parsed[k] = float(v)
            except ValueError:
                parsed[k] = None
        elif k in NUMERIC_INT_KEYS:
            try:
                parsed[k] = int(float(v))
            except ValueError:
                parsed[k] = None
        else:
            parsed[k] = v
    return parsed


def load_annotations(synth_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load annotations.csv written by the synthetic generator and
    return a dict keyed by the RGB path RELATIVE to synth_root.

    i.e. row["rgb_path"] typically looks like:
      'apriltag_36h11_distance/rgb_apriltag_36h11_distance_d030.png'
    """
    csv_path = synth_root / "annotations.csv"
    meta: Dict[str, Dict[str, Any]] = {}

    if not csv_path.is_file():
        print(f"[WARN] annotations.csv not found at {csv_path}, "
              f"JSON will not include generator metadata.")
        return meta

    print(f"[INFO] Loading annotations from: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "rgb_path" not in row:
                continue
            key = row["rgb_path"].replace("\\", "/")
            if not key:
                continue

            parsed = _parse_numeric_fields(row)
            meta[key] = parsed

    print(f"[INFO] Loaded {len(meta)} annotation rows.")
    return meta

# -------------------------------------------------------------------
# Main evaluation logic
# -------------------------------------------------------------------

def main():
    synthetic_root = find_synthetic_root()
    if synthetic_root is None:
        print("[ERROR] synthetic_markers directory not found anywhere under repo root.")
        sys.exit(1)

    print(f"[INFO] Synthetic root: {synthetic_root}")

    # Load annotations (if present)
    annotations = load_annotations(synthetic_root)

    # Collect all RGB images
    rgb_images: List[Path] = sorted(synthetic_root.rglob("rgb_*.png"))
    print(f"[INFO] Found {len(rgb_images)} RGB images")

    if not rgb_images:
        print("[WARN] No rgb_*.png images found. JSON will be empty.")

    # Create detector manager (no OAK device, synthetic only)
    manager = DetectorManager()

    evaluations: List[Dict[str, Any]] = []

    for idx, img_path in enumerate(rgb_images, start=1):
        # Path relative to repo root for nice reporting
        try:
            rel_path_repo = img_path.relative_to(REPO_ROOT)
        except ValueError:
            rel_path_repo = img_path

        # Path relative to synthetic_root to match annotations.csv rgb_path
        try:
            rel_path_synth = img_path.relative_to(synthetic_root)
        except ValueError:
            rel_path_synth = img_path

        rel_synth_str = str(rel_path_synth).replace("\\", "/")
        meta = annotations.get(rel_synth_str, {})

        marker_type = infer_marker_type_from_name(img_path)
        if marker_type is None:
            print(f"[WARN] Could not infer marker type for {rel_path_repo}, skipping.")
            continue

        # Load RGB image
        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[WARN] Could not read image {rel_path_repo}, skipping.")
            continue

        # Dummy depth (synthetic eval)
        depth = np.zeros(rgb.shape[:2], dtype=np.uint16)

        print(f"[INFO] [{idx}/{len(rgb_images)}] Processing {rel_path_repo} "
              f"(marker_type={marker_type})")

        t0 = time.perf_counter()
        result = manager.detect_marker(marker_type, rgb, depth)
        print(
            f"  -> detected={result.get('detected')} "
            f"id={result.get('marker_id')} "
            f"decoded={result.get('decoded')} "
            f"conf={result.get('confidence')}"
        )

        if "total_detection_time_ms" not in result:
            total_ms = (time.perf_counter() - t0) * 1000.0
        else:
            total_ms = result["total_detection_time_ms"]

        # Base entry (pure detector output)
        eval_entry: Dict[str, Any] = {
            "image": str(rel_path_repo),
            "marker_type": marker_type,  # type passed to detector
            "detected": bool(result.get("detected", False)),
            "marker_id": result.get("marker_id", None),
            "decoded": result.get("decoded", None),
            "confidence": result.get("confidence", None),
            "error": result.get("error", None),
            "total_detection_time_ms": float(total_ms),
        }

        # Attach generator metadata if we have it
        if meta:
            # distinguish GT marker vs detected marker
            gt_marker_id = meta.get("marker_id", None)
            gt_marker_type = meta.get("marker_type", None)

            eval_entry.update({
                "sample_id": meta.get("sample_id"),
                "split": meta.get("split"),
                "gt_marker_type": gt_marker_type,
                "gt_marker_id": gt_marker_id,
                "marker_text": meta.get("marker_text"),
                "distance_m": meta.get("distance_m"),
                "x_deg": meta.get("x_deg"),
                "y_deg": meta.get("y_deg"),
                "occlusion_pct": meta.get("occlusion_pct"),
                "lighting": meta.get("lighting"),
                "qr_size_m": meta.get("qr_size_m"),
                "fx": meta.get("fx"),
                "fy": meta.get("fy"),
                "cx": meta.get("cx"),
                "cy": meta.get("cy"),
                "baseline_m": meta.get("baseline_m"),
                "img_width": meta.get("img_width"),
                "img_height": meta.get("img_height"),
                # also store the rgb_path key used for lookup
                "rgb_path_rel_synthetic_root": rel_synth_str,
            })

        evaluations.append(eval_entry)

    print(f"\n[INFO] Total evaluated images: {len(evaluations)}")

    # Save JSON
    out_dir = THIS_FILE.parent / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_opencv_eval.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2)

    print(f"[INFO] Saved evaluation to: {out_path}")


if __name__ == "__main__":
    main()
