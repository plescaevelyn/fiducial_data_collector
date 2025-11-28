#!/usr/bin/env python3
import json
import sys
import csv
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

# Default locations (same style as your other scripts)
DEFAULT_DATASETS_DIR = THIS_FILE.parent / "datasets"
DEFAULT_JSON = DEFAULT_DATASETS_DIR / "synthetic_opencv_eval.json"
DEFAULT_OUT_CSV = DEFAULT_DATASETS_DIR / "synthetic_opencv_eval_summary.csv"


def load_eval_json(path: Path):
    if not path.is_file():
        print(f"[ERROR] JSON file not found: {path}")
        sys.exit(1)
    print(f"[INFO] Loading eval JSON from: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} entries.")
    return data


def infer_param_name_and_value(entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    """
    Decide which parameter we are varying in this split:

    - *_distance   -> distance_m
    - *_x_rotation -> x_deg
    - *_y_rotation -> y_deg
    - *_occlusion  -> occlusion_pct
    - *_lighting   -> lighting
    """
    split = entry.get("split", "") or ""
    split = str(split)

    if split.endswith("_distance"):
        return "distance_m", entry.get("distance_m")
    if split.endswith("_x_rotation"):
        return "x_deg", entry.get("x_deg")
    if split.endswith("_y_rotation"):
        return "y_deg", entry.get("y_deg")
    if split.endswith("_occlusion"):
        return "occlusion_pct", entry.get("occlusion_pct")
    if split.endswith("_lighting"):
        return "lighting", entry.get("lighting")

    # Fallback: no specific parameter
    return None, None


def marker_family(entry: Dict[str, Any]) -> str:
    """
    Use gt_marker_type if available (qr, apriltag_36h11, etc.).
    Otherwise fall back to detector marker_type (qr_code, aruco_4x4_50, ...).
    """
    gt = entry.get("gt_marker_type")
    if gt is not None and gt != "":
        return str(gt)
    return str(entry.get("marker_type", "unknown"))


def summarize(data, out_csv: Path):
    """
    Aggregate per (marker_family, split, param_name, param_value)
    -> detection rate, error rate, mean time.
    """
    groups: Dict[Tuple[str, str, Optional[str], Optional[str]], Dict[str, Any]] = {}

    for e in data:
        fam = marker_family(e)
        split = str(e.get("split", "") or "")
        param_name, param_value = infer_param_name_and_value(e)

        # normalize param_value to string for grouping CSV-friendly
        if isinstance(param_value, float):
            key_param_value = f"{param_value:.6g}"  # compact decimal
        else:
            key_param_value = str(param_value) if param_value is not None else ""

        key = (fam, split, param_name, key_param_value)

        g = groups.get(key)
        if g is None:
            g = {
                "marker_family": fam,
                "split": split,
                "param_name": param_name,
                "param_value": param_value,
                "num_samples": 0,
                "num_detected": 0,
                "num_error": 0,
                "sum_time_ms": 0.0,
            }
            groups[key] = g

        g["num_samples"] += 1
        if e.get("detected"):
            g["num_detected"] += 1
        if e.get("error") not in (None, ""):
            g["num_error"] += 1
        t_ms = e.get("total_detection_time_ms")
        if isinstance(t_ms, (int, float)):
            g["sum_time_ms"] += float(t_ms)

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing summary CSV to: {out_csv}")

    fieldnames = [
        "marker_family",
        "split",
        "param_name",
        "param_value",
        "num_samples",
        "num_detected",
        "num_error",
        "detection_rate",
        "error_rate",
        "mean_time_ms",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # stable-ish ordering
        for key in sorted(groups.keys()):
            g = groups[key]
            n = g["num_samples"]
            detected = g["num_detected"]
            errors = g["num_error"]
            detection_rate = detected / n if n > 0 else 0.0
            error_rate = errors / n if n > 0 else 0.0
            mean_time_ms = g["sum_time_ms"] / n if n > 0 else 0.0

            writer.writerow(dict(
                marker_family=g["marker_family"],
                split=g["split"],
                param_name=g["param_name"],
                param_value=g["param_value"],
                num_samples=n,
                num_detected=detected,
                num_error=errors,
                detection_rate=detection_rate,
                error_rate=error_rate,
                mean_time_ms=mean_time_ms,
            ))

    print("[INFO] Summary CSV generated.")


def main():
    # Optional CLI: summarize_synthetic_eval.py [eval_json] [out_csv]
    json_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_JSON
    out_csv = Path(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_OUT_CSV

    data = load_eval_json(json_path)
    summarize(data, out_csv)


if __name__ == "__main__":
    main()
