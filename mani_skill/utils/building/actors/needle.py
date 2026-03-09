"""
Needle Actor for ManiSkill
Based on DexMimicGen's needle object for threading tasks

Place this file in: mani_skill/utils/building/actors/needle.py
"""

import numpy as np
import sapien
from sapien import Pose
from typing import Optional
from mani_skill.utils.building.actor_builder import ActorBuilder

def build_needle(
    scene: sapien.Scene,
    name: str = "needle",
    length: float = 0.12,
    shaft_radius: float = 0.001,
    tip_length: float = 0.01,
    density: float = 8000.0,
    color: Optional[np.ndarray] = None,
    initial_pose: Optional[Pose] = None,
    return_builder: bool = False,
) -> ActorBuilder | sapien.Entity:
    """
    Build a needle actor using box primitives (shaft + tip).
    
    Args:
        scene: SAPIEN scene to add the actor to
        name: Name of the actor
        length: Total length of the needle (m)
        shaft_radius: Half thickness of the needle shaft (m)
        tip_length: Length of the pointed tip section (m)
        density: Material density (kg/m³), defaults to steel-like
        color: RGBA color array, defaults to metallic gray
        initial_pose: Initial pose of the actor
        
    Returns:
        The built SAPIEN actor entity
    """
    if color is None:
        color = np.array([0.6, 0.6, 0.65, 1.0])  # Metallic gray
    
    builder = scene.create_actor_builder()
    
    # Material for rendering (metallic appearance)
    material = sapien.render.RenderMaterial(
        base_color=color,
        metallic=0.8,
        roughness=0.3,
        specular=0.5,
    )
    
    # Physical material
    physical_material = sapien.physx.PhysxMaterial(
        static_friction=0.3,
        dynamic_friction=0.3,
        restitution=0.05,
    )
    
    # Geometry is aligned along +X.
    # We place the base at local x=0 so "tip at x=length" matches task logic.
    tip_length = float(np.clip(tip_length, 0.0, length))
    shaft_length = float(max(length - tip_length, 0.0))

    # Shaft (rectangular prism)
    shaft_pose = sapien.Pose(p=[shaft_length / 2, 0, 0])
    shaft_half_size = [shaft_length / 2, shaft_radius, shaft_radius]
    builder.add_box_visual(
        pose=shaft_pose,
        half_size=shaft_half_size,
        material=material,
    )
    builder.add_box_collision(
        pose=shaft_pose,
        half_size=shaft_half_size,
        material=physical_material,
        density=density,
    )

    # Tip (smaller prism)
    tip_pose = sapien.Pose(p=[length - tip_length / 2, 0, 0])
    tip_half_size = [tip_length / 2, shaft_radius * 0.6, shaft_radius * 0.6]
    builder.add_box_visual(
        pose=tip_pose,
        half_size=tip_half_size,
        material=material,
    )
    builder.add_box_collision(
        pose=tip_pose,
        half_size=tip_half_size,
        material=physical_material,
        density=density,
    )

    # Set initial pose (horizontal, along x-axis by default)
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, 0])
    builder.initial_pose = initial_pose
    
    # Build and return the actor
    if return_builder:
        return builder
    return builder.build(name=name)