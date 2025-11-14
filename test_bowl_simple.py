#!/usr/bin/env python3
"""Simple test to verify the ceramic bowl file exists and can be referenced"""

from pathlib import Path
from mani_skill import PACKAGE_ASSET_DIR

# Check if the bowl file was copied correctly
bowl_path = PACKAGE_ASSET_DIR / "dish_into_rack/ceramic_bowl.obj"

print(f"Bowl path: {bowl_path}")
print(f"Bowl file exists: {bowl_path.exists()}")

if bowl_path.exists():
    file_size = bowl_path.stat().st_size / (1024 * 1024)  # MB
    print(f"Bowl file size: {file_size:.2f} MB")
    print("\n✓ Ceramic bowl file is ready!")
else:
    print("\n✗ Bowl file not found!")

# Also print the environment file location
env_file = Path(__file__).parent / "mani_skill/envs/tasks/tabletop/place_dish_in_rack.py"
print(f"\nEnvironment file: {env_file}")
print(f"Environment file exists: {env_file.exists()}")
