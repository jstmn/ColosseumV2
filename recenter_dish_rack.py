"""Recenter the dish rack STL so its origin is at the actual center."""

import numpy as np
from stl import mesh
import shutil
from pathlib import Path

# Paths
stl_path = Path("mani_skill/assets/dish_into_rack/dish_rack_with_connectors.stl")
backup_path = stl_path.with_suffix(".stl.backup")

# Backup the original
if not backup_path.exists():
    shutil.copy(stl_path, backup_path)
    print(f"Created backup at {backup_path}")
else:
    print(f"Backup already exists at {backup_path}")

# Load the STL
print(f"\nLoading {stl_path}...")
dish_rack_mesh = mesh.Mesh.from_file(str(stl_path))

# Get all vertices
vertices = dish_rack_mesh.vectors.reshape(-1, 3)

# Calculate bounding box
min_bounds = vertices.min(axis=0)
max_bounds = vertices.max(axis=0)
center = (min_bounds + max_bounds) / 2

print(f"\nOriginal bounds:")
print(f"  Min: {min_bounds}")
print(f"  Max: {max_bounds}")
print(f"  Center: {center}")
print(f"  Size: {max_bounds - min_bounds}")

# Translate all vertices to center the mesh
print(f"\nTranslating mesh by {-center}...")
dish_rack_mesh.vectors -= center

# Verify the new bounds
new_vertices = dish_rack_mesh.vectors.reshape(-1, 3)
new_min = new_vertices.min(axis=0)
new_max = new_vertices.max(axis=0)
new_center = (new_min + new_max) / 2

print(f"\nNew bounds:")
print(f"  Min: {new_min}")
print(f"  Max: {new_max}")
print(f"  Center: {new_center}")

# Save the recentered mesh
print(f"\nSaving recentered mesh to {stl_path}...")
dish_rack_mesh.save(str(stl_path))

print("\n✓ Done! The dish rack is now centered.")
print(f"\nThe offset that was being applied in code: [-0.12606856536, -0.10041320611, 0.0]")
print(f"The actual center offset was: {center}")
print(f"\nYou can now remove the _rack_mesh_offset from the code.")
