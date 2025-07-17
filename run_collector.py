#!/usr/bin/env python3
"""
Main entry script for the Fiducial Data Collector.
Initializes and runs the guided data collection process.
"""

import sys
import os
from pathlib import Path

# Add local module path
sys.path.append(str(Path(__file__).parent))

from main_collector import FiducialDataCollector
from config.test_configurations import get_test_summary

def main():
    print("FIDUCIAL DATA COLLECTOR")
    print("=" * 50)
    print("Systematic data collection for fiducial markers")
    print("Inspired by CopperTag methodology, optimized for OAK-D Lite")

    # Display test summary
    summary = get_test_summary()
    print("\nTEST SUMMARY:")
    print(f"   Markers: 7 types")
    print(f"   Distance tests: {summary['distance_tests']}")
    print(f"   Rotation tests: {summary['rotation_x_tests'] + summary['rotation_y_tests'] + summary['rotation_z_tests']}")
    print(f"   Occlusion tests: {summary['occlusion_tests']}")
    print(f"   Lighting tests: {summary['lighting_tests']}")
    print(f"   TOTAL: {summary['total_tests']} tests")
    print(f"   Estimated time: {summary['estimated_hours']} hours")

    # User confirmation
    print("\nREADY FOR COLLECTION?")
    print("   Make sure you have:")
    print("   - OAK-D Lite connected")
    print("   - Printed markers (5cm x 5cm)")
    print("   - Ruler for distance measurement")
    print("   - Variable light sources")
    print("   - Occlusion objects")

    response = input("\nProceed with data collection? (y/n): ").lower().strip()

    if response not in ['y', 'yes']:
        print("Collection cancelled by user.")
        return

    try:
        collector = FiducialDataCollector()
        collector.run_data_collection()

    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
        print("Partial data has been saved.")

    except Exception as e:
        print(f"\nError during collection: {e}")
        print("Please contact the developer for support.")

    finally:
        print("\nCollection session finished.")
        print("Check the datasets/ folder for results.")

if __name__ == "__main__":
    main()