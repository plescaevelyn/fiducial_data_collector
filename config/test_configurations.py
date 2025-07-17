# Optimized Test Configurations for a Single Operator
# Inspired by the CopperTag methodology but adapted for real-world measurements

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class TestType(Enum):
    DISTANCE = "distance"
    ROTATION_X = "rotation_x"
    ROTATION_Y = "rotation_y"
    ROTATION_Z = "rotation_z"
    OCCLUSION = "occlusion"
    LIGHTING = "lighting"

class MarkerType(Enum):
    # Available OpenCV types (implemented immediately)
    ARUCO_4X4_50 = "aruco_4x4_50"
    ARUCO_6X6_250 = "aruco_6x6_250"
    APRILTAG_36H11 = "apriltag_36h11"
    QR_CODE = "qr_code"

    # External markers (to be implemented)
    RUNETAG = "runetag"
    CHROMATAG = "chromatag"
    COPPERTAG = "coppertag"

@dataclass
class TestConfiguration:
    test_type: TestType
    marker_type: MarkerType
    parameter_value: float
    parameter_unit: str
    description: str
    estimated_duration_seconds: int
    special_instructions: str

# Optimized configurations (inspired by CopperTag)
DISTANCE_TESTS = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # 8 distances
ROTATION_X_TESTS = [-60, -40, -20, 0, 20, 40, 60]  # 7 angles
ROTATION_Y_TESTS = [-60, -40, -20, 0, 20, 40, 60]  # 7 angles
ROTATION_Z_TESTS = [0, 45, 90, 180, 270]  # 5 angles
OCCLUSION_TESTS = [5, 10, 15, 20]  # 4 levels (%)
LIGHTING_CONDITIONS = ["bright", "normal", "dim", "shadow"]  # 4 conditions

def generate_all_test_configurations() -> List[TestConfiguration]:
    """Generates all test configurations (280 total tests)."""
    configurations = []
    available_markers = list(MarkerType)

    for marker in available_markers:
        for distance in DISTANCE_TESTS:
            config = TestConfiguration(
                test_type=TestType.DISTANCE,
                marker_type=marker,
                parameter_value=distance,
                parameter_unit="m",
                description=f"Test {marker.value} at {distance}m",
                estimated_duration_seconds=90,
                special_instructions=f"Place the marker {distance}m from the camera"
            )
            configurations.append(config)

    for marker in available_markers:
        for angle in ROTATION_X_TESTS:
            config = TestConfiguration(
                test_type=TestType.ROTATION_X,
                marker_type=marker,
                parameter_value=angle,
                parameter_unit="degrees",
                description=f"Test {marker.value} X rotation {angle}°",
                estimated_duration_seconds=75,
                special_instructions=f"Tilt the camera by {angle}° on the X-axis"
            )
            configurations.append(config)

    for marker in available_markers:
        for angle in ROTATION_Y_TESTS:
            config = TestConfiguration(
                test_type=TestType.ROTATION_Y,
                marker_type=marker,
                parameter_value=angle,
                parameter_unit="degrees",
                description=f"Test {marker.value} Y rotation {angle}°",
                estimated_duration_seconds=75,
                special_instructions=f"Pan the camera by {angle}° on the Y-axis"
            )
            configurations.append(config)

    for marker in available_markers:
        for angle in ROTATION_Z_TESTS:
            config = TestConfiguration(
                test_type=TestType.ROTATION_Z,
                marker_type=marker,
                parameter_value=angle,
                parameter_unit="degrees",
                description=f"Test {marker.value} Z rotation {angle}°",
                estimated_duration_seconds=60,
                special_instructions=f"Rotate the marker by {angle}°"
            )
            configurations.append(config)

    for marker in available_markers:
        for occlusion in OCCLUSION_TESTS:
            config = TestConfiguration(
                test_type=TestType.OCCLUSION,
                marker_type=marker,
                parameter_value=occlusion,
                parameter_unit="%",
                description=f"Test {marker.value} with {occlusion}% occlusion",
                estimated_duration_seconds=90,
                special_instructions=f"Cover {occlusion}% of the marker"
            )
            configurations.append(config)

    for marker in available_markers:
        for lighting in LIGHTING_CONDITIONS:
            config = TestConfiguration(
                test_type=TestType.LIGHTING,
                marker_type=marker,
                parameter_value=0,
                parameter_unit="condition",
                description=f"Test {marker.value} in {lighting} lighting",
                estimated_duration_seconds=75,
                special_instructions=_get_lighting_instructions(lighting)
            )
            configurations.append(config)

    return configurations

def _get_lighting_instructions(lighting_condition: str) -> str:
    instructions = {
        "bright": "Direct strong light",
        "normal": "Normal room lighting",
        "dim": "Low light",
        "shadow": "Partial shadow over the marker"
    }
    return instructions.get(lighting_condition, "Unknown condition")

def get_test_summary() -> Dict[str, int]:
    """Test summary: 245 total tests in ~7 hours (7 markers × 35 tests)."""
    return {
        "distance_tests": 56,
        "rotation_x_tests": 49,
        "rotation_y_tests": 49,
        "rotation_z_tests": 35,
        "occlusion_tests": 28,
        "lighting_tests": 28,
        "total_tests": 245,
        "estimated_hours": 7
    }
