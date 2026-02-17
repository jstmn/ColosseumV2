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
    eye_radius: float = 0.003,
    eye_distance_from_end: float = 0.015,
    density: float = 8000.0,
    color: Optional[np.ndarray] = None,
    initial_pose: Optional[Pose] = None,
    return_builder: bool = False,
) -> ActorBuilder | sapien.Entity:
    """
    Build a needle actor - a thin cylindrical shaft with a pointed tip and an eye for threading.
    
    Args:
        scene: SAPIEN scene to add the actor to
        name: Name of the actor
        length: Total length of the needle (m)
        shaft_radius: Radius of the needle shaft (m)
        tip_length: Length of the pointed tip section (m)
        eye_radius: Radius of the eye hole (m)
        eye_distance_from_end: Distance of eye center from the blunt end (m)
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
    
    # Build the main shaft (excluding tip and eye region)
    # We'll split the needle into sections to accommodate the eye
    
    # Section 1: From blunt end to before eye
    section1_length = eye_distance_from_end - eye_radius - 0.001
    section1_center = section1_length / 2
    
    if section1_length > 0:
        pose1 = sapien.Pose(p=[section1_center, 0, 0])
        
        builder.add_cylinder_visual(
            pose=pose1,
            radius=shaft_radius,
            half_length=section1_length / 2,
            material=material,
        )
        
        builder.add_cylinder_collision(
            pose=pose1,
            radius=shaft_radius,
            half_length=section1_length / 2,
            material=physical_material,
            density=density,
        )
    
    # Eye region: Create a ring shape (approximated with cylinders)
    # For simplicity, we'll create the eye as a visual feature
    # and use two short shaft segments around it for collision
    eye_center_x = eye_distance_from_end
    
    # Small segments on either side of the eye
    segment_length = 0.002
    
    for offset in [-eye_radius - segment_length/2, eye_radius + segment_length/2]:
        pose_eye_segment = sapien.Pose(p=[eye_center_x + offset, 0, 0])
        
        builder.add_cylinder_visual(
            pose=pose_eye_segment,
            radius=shaft_radius,
            half_length=segment_length / 2,
            material=material,
        )
        
        builder.add_cylinder_collision(
            pose=pose_eye_segment,
            radius=shaft_radius,
            half_length=segment_length / 2,
            material=physical_material,
            density=density,
        )
    
    # Section 2: From after eye to before tip
    section2_start = eye_distance_from_end + eye_radius + 0.001
    section2_end = length - tip_length
    section2_length = section2_end - section2_start
    section2_center = (section2_start + section2_end) / 2
    
    if section2_length > 0:
        pose2 = sapien.Pose(p=[section2_center, 0, 0])
        
        builder.add_cylinder_visual(
            pose=pose2,
            radius=shaft_radius,
            half_length=section2_length / 2,
            material=material,
        )
        
        builder.add_cylinder_collision(
            pose=pose2,
            radius=shaft_radius,
            half_length=section2_length / 2,
            material=physical_material,
            density=density,
        )
    
    # Tip: Tapered cone shape pointing in +x direction
    # Taper effect approximated with multiple cylinders of decreasing radius
    num_tip_segments = 3
    for i in range(num_tip_segments):
        seg_start = length - tip_length + (i * tip_length / num_tip_segments)
        seg_end = seg_start + tip_length / num_tip_segments
        seg_center = (seg_start + seg_end) / 2
        
        # Linearly decrease radius toward tip
        radius_factor = 1.0 - (i + 1) / num_tip_segments
        seg_radius = shaft_radius * max(radius_factor, 0.3)
        
        seg_pose = sapien.Pose(p=[seg_center, 0, 0])
        
        builder.add_cylinder_visual(
            pose=seg_pose,
            radius=seg_radius,
            half_length=tip_length / num_tip_segments / 2,
            material=material,
        )
        
        builder.add_cylinder_collision(
            pose=seg_pose,
            radius=seg_radius,
            half_length=tip_length / num_tip_segments / 2,
            material=physical_material,
            density=density,
        )
    
    # Visual representation of the eye (torus or ring)
    # Create a ring using small spheres for visual clarity
    num_eye_points = 8
    for i in range(num_eye_points):
        angle = i * 2 * np.pi / num_eye_points
        y = eye_radius * np.cos(angle)
        z = eye_radius * np.sin(angle)
        
        eye_point_pose = sapien.Pose(p=[eye_center_x, y, z])
        
        builder.add_sphere_visual(
            pose=eye_point_pose,
            radius=shaft_radius * 1.2,
            material=material,
        )
    
    # Set initial pose (horizontal, along x-axis by default)
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, 0])
    builder.initial_pose = initial_pose
    
    # Build and return the actor
    if return_builder:
        return builder
    return builder.build(name=name)