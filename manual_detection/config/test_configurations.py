# config/test_configurations.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

# ---------- Enums expected by main_collector.py ----------
class MarkerType(str, Enum):
    QR_CODE        = "qr_code"
    APRILTAG_36H11 = "apriltag_36h11"   # AprilTag-3
    # APRILTAG_25H9  = "apriltag_25h9"    # AprilTag-2 (extra)
    # ARUCO_4X4_50   = "aruco_4x4_50"
    # ARUCO_6X6_250  = "aruco_6x6_250"
    # RUNETAG        = "runetag"
    # CHROMATAG      = "chromatag"
    # COPPERTAG      = "coppertag"

    # Keep AprilTag-2 (25h9) in plan & scalable, even if not in the 7 “representative”:

class TestType(str, Enum):
    DISTANCE   = "distance"
    ROTATION_X = "rotation_x"
    ROTATION_Y = "rotation_y"
    ROTATION_Z = "rotation_z"
    OCCLUSION  = "occlusion"
    LIGHTING   = "lighting"

# ---------- Representative 7 markers (order from your doc) ----------
REPRESENTATIVE_MARKERS: List[MarkerType] = [
    MarkerType.QR_CODE,
    MarkerType.APRILTAG_36H11,
    # MarkerType.APRILTAG_25H9,
    # MarkerType.ARUCO_4X4_50,
    # MarkerType.ARUCO_6X6_250,
    # MarkerType.RUNETAG,
    # MarkerType.CHROMATAG,
    # MarkerType.COPPERTAG,
]

# Default IDs per family (QR / non-id systems use None)
DEFAULT_MARKER_ID: Dict[MarkerType, Optional[int]] = {
    MarkerType.QR_CODE:        None,
    MarkerType.APRILTAG_36H11: 0,
    # MarkerType.APRILTAG_25H9:  0,    # AT2 rămâne prezent și scalat
    # MarkerType.ARUCO_4X4_50:   18,   # schimbă după ID-ul printat de tine
    # MarkerType.ARUCO_6X6_250:  18,   # idem
    # MarkerType.RUNETAG:        None,
    # MarkerType.CHROMATAG:      None,
    # MarkerType.COPPERTAG:      None,
}

@dataclass
class TestConfiguration:
    description: str
    marker_type: MarkerType
    marker_id: Optional[int]
    type: TestType
    param: Optional[float]        # meters (distance), degrees (rot), ratio (occlusion), gain (lighting)
    estimated_seconds: int = 90

    # Backward-compat: unele locuri folosesc .test_type
    @property
    def test_type(self) -> TestType:
        return self.type

# ---------- Sets from your README-spec ----------
# DISTANCE (8): 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
DISTANCES_M: List[float] = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]  # 7

# ROTATION X/Y (7): -60, -40, -20, 0, 20, 40, 60
ROT_AXIS_DEG: List[float] = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]  # 7

# ROTATION Z (5): 0, 45, 90, 180, 270
ROT_Z_DEG: List[float] = [0.0, 45.0, 90.0, 180.0, 270.0]  # 5

# OCCLUSION (4): 5%, 10%, 15%, 20%
OCCLUSION_RATIO: List[float] = [0.05, 0.10, 0.15, 0.20]  # 4

# LIGHTING (4): Bright, Normal, Dim, Shadow — map to gains (float) but păstrăm eticheta în description
LIGHTING_LABEL_TO_GAIN: Dict[str, float] = {
    "Bright": 1.40,
    "Normal": 1.00,
    "Dim":    0.80,
    "Shadow": 0.60,
}
LIGHTING_SEQUENCE: List[str] = ["Bright", "Normal", "Dim", "Shadow"]  # 4

def _mk(m: MarkerType, mid: Optional[int], t: TestType, val: float, label: Optional[str] = None) -> TestConfiguration:
    if t == TestType.LIGHTING and label is not None:
        desc = f"{m.value} lighting={label} (gain={val})"
    elif t == TestType.OCCLUSION:
        desc = f"{m.value} occlusion={int(val*100)}%"
    elif t in (TestType.ROTATION_X, TestType.ROTATION_Y, TestType.ROTATION_Z):
        desc = f"{m.value} {t.value}={val}°"
    elif t == TestType.DISTANCE:
        desc = f"{m.value} distance={val}m"
    else:
        desc = f"{m.value} {t.value}={val}"
    return TestConfiguration(
        description=desc,
        marker_type=m,
        marker_id=mid,
        type=t,
        param=val,
        estimated_seconds=90,
    )

def _generate_marker_block(m: MarkerType) -> List[TestConfiguration]:
    """Build exactly 35 tests per marker: 8 dist + 7 rx + 7 ry + 5 rz + 4 occ + 4 light."""
    mid = DEFAULT_MARKER_ID[m]
    block: List[TestConfiguration] = []
    # Distance (8)
    for d in DISTANCES_M:
        block.append(_mk(m, mid, TestType.DISTANCE, d))
    # Rotation X (7)
    for a in ROT_AXIS_DEG:
        block.append(_mk(m, mid, TestType.ROTATION_X, a))
    # Rotation Y (7)
    for a in ROT_AXIS_DEG:
        block.append(_mk(m, mid, TestType.ROTATION_Y, a))
    # Rotation Z (5)
    for a in ROT_Z_DEG:
        block.append(_mk(m, mid, TestType.ROTATION_Z, a))
    # Occlusion (4)
    for occ in OCCLUSION_RATIO:
        block.append(_mk(m, mid, TestType.OCCLUSION, occ))
    # Lighting (4) using label -> gain
    for lab in LIGHTING_SEQUENCE:
        gain = LIGHTING_LABEL_TO_GAIN[lab]
        block.append(_mk(m, mid, TestType.LIGHTING, gain, label=lab))
    return block

def generate_full_245_plan() -> List[TestConfiguration]:
    """
    7 representative markers × 35 tests/marker = 245 tests (exact spec din README).
    """
    plan: List[TestConfiguration] = []
    for m in REPRESENTATIVE_MARKERS:
        plan.extend(_generate_marker_block(m))
    return plan

def generate_full_280_plan_including_at2() -> List[TestConfiguration]:
    """
    Extended: include AprilTag-2 (25h9) în plus: 8×35 = 280.
    """
    plan: List[TestConfiguration] = []
    return plan

# --- Ce importă main_collector.py / run_collector.py ---
def generate_all_test_configurations() -> List[TestConfiguration]:
    # Dacă vrei EXACT 245 ca în README:
    return generate_full_245_plan()
    # Dacă vrei 280 cu AprilTag-2 inclus full:
    # return generate_full_280_plan_including_at2()

def get_test_summary() -> Dict:
    plan = generate_all_test_configurations()

    distance_tests   = sum(1 for t in plan if t.type == TestType.DISTANCE)
    rotation_x_tests = sum(1 for t in plan if t.type == TestType.ROTATION_X)
    rotation_y_tests = sum(1 for t in plan if t.type == TestType.ROTATION_Y)
    rotation_z_tests = sum(1 for t in plan if t.type == TestType.ROTATION_Z)
    occlusion_tests  = sum(1 for t in plan if t.type == TestType.OCCLUSION)
    lighting_tests   = sum(1 for t in plan if t.type == TestType.LIGHTING)

    total_seconds = sum(t.estimated_seconds for t in plan)
    estimated_hours = round(total_seconds / 3600.0, 1)

    return {
        "total_tests": len(plan),
        "distance_tests": distance_tests,
        "rotation_x_tests": rotation_x_tests,
        "rotation_y_tests": rotation_y_tests,
        "rotation_z_tests": rotation_z_tests,
        "occlusion_tests": occlusion_tests,
        "lighting_tests": lighting_tests,
        "estimated_hours": estimated_hours,
        "marker_types": [m.value for m in REPRESENTATIVE_MARKERS],
        "num_marker_types": len(REPRESENTATIVE_MARKERS),
    }

# --- Compatibility aliases (other modules may import these) ---
def generate_test_plan():
    # Alias vechi — direcționează spre planul oficial din README
    return generate_full_245_plan()
