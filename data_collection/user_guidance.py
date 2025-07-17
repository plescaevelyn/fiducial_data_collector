# Interactive user guidance system
# Provides step-by-step instructions for each test type

import time
from typing import Dict, List
from config.test_configurations import TestType, MarkerType, TestConfiguration

class UserGuidanceSystem:
    """
    Guidance system providing detailed instructions to the user.
    Ensures repeatability and accuracy of measurements.
    """

    def __init__(self):
        self.current_setup = None
        self.setup_history = []

    def show_test_setup_instructions(self, config: TestConfiguration) -> bool:
        """
        Displays detailed setup instructions for a test.

        Returns:
            True if the user confirms setup, False to skip
        """
        print("\n" + "SETUP INSTRUCTIONS" + "=" * 45)

        self._show_marker_instructions(config.marker_type)

        if config.test_type == TestType.DISTANCE:
            return self._show_distance_instructions(config)
        elif config.test_type == TestType.ROTATION_X:
            return self._show_rotation_x_instructions(config)
        elif config.test_type == TestType.ROTATION_Y:
            return self._show_rotation_y_instructions(config)
        elif config.test_type == TestType.ROTATION_Z:
            return self._show_rotation_z_instructions(config)
        elif config.test_type == TestType.OCCLUSION:
            return self._show_occlusion_instructions(config)
        elif config.test_type == TestType.LIGHTING:
            return self._show_lighting_instructions(config)

        return True

    def _show_marker_instructions(self, marker_type: MarkerType):
        """Displays instructions for preparing the marker."""
        marker_info = {
            MarkerType.ARUCO_4X4_50: {
                'name': 'ArUco 4x4_50',
                'file': 'markers/aruco/aruco_4x4_50_id_0.pdf',
                'description': 'Standard industrial ArUco marker'
            },
            MarkerType.ARUCO_6X6_250: {
                'name': 'ArUco 6x6_250', 
                'file': 'markers/aruco/aruco_6x6_250_id_0.pdf',
                'description': 'Improved precision ArUco marker'
            },
            MarkerType.APRILTAG_36H11: {
                'name': 'AprilTag 36h11',
                'file': 'markers/apriltag/apriltag_36h11_id_0.pdf',
                'description': 'Most robust AprilTag version'
            },
            MarkerType.QR_CODE: {
                'name': 'QR Code',
                'file': 'markers/qr/qr_test_data.pdf',
                'description': 'Standard QR Code'
            }
        }

        info = marker_info.get(marker_type, {'name': marker_type.value, 'file': 'N/A', 'description': 'Experimental marker'})

        print(f"MARKER: {info['name']}")
        print(f"File: {info['file']}")
        print(f"Description: {info['description']}")
        print("Print size: 5cm x 5cm")
        print("Marker preparation:")
        print("   1. Print the marker at exact size (5cm x 5cm)")
        print("   2. Use white matte paper (not glossy)")
        print("   3. Mount the marker on a flat and rigid surface")
        print("   4. Ensure the marker is not folded or damaged")

    def _confirm_setup(self) -> bool:
        """Asks the user to confirm the setup."""
        print("\nSETUP CONFIRMATION:")
        print("   - Is the marker positioned correctly? ✓")
        print("   - Is the camera configured as instructed? ✓")
        print("   - Is the marker fully visible in the image? ✓")
        print("   - Does the setup meet test requirements? ✓")

        while True:
            response = input("\nSetup complete? (y/n/s to skip): ").lower().strip()

            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                print("Please adjust the setup accordingly.")
                return False
            elif response in ['s', 'skip']:
                print("Test skipped at user request.")
                return False
            else:
                print("Please respond with 'y', 'n' or 's'")