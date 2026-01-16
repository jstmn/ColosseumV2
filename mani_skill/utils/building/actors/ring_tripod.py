"""
Ring Tripod Actor for ManiSkill
Based on DexMimicGen's ring_tripod composite object

Place this file in: mani_skill/utils/building/actors/ring_tripod.py
"""

import numpy as np
import sapien
from sapien import Pose
from typing import Optional


def build_ring_tripod(
    scene: sapien.Scene,
    name: str = "ring_tripod",
    base_size: float = 0.15,
    base_thickness: float = 0.01,
    pole_height: float = 0.12,
    pole_radius: float = 0.008,
    ring_radius: float = 0.025,
    ring_thickness: float = 0.01,
    density: float = 1000.0,
    color: Optional[np.ndarray] = None,
    initial_pose: Optional[Pose] = None,
) -> sapien.Entity:
    """
    Build a ring tripod actor - a square base with a vertical pole and a ring at the top.
    Used for needle threading tasks.
    
    Args:
        scene: SAPIEN scene to add the actor to
        name: Name of the actor
        base_size: Side length of the square base (m)
        base_thickness: Thickness/height of the base (m)
        pole_height: Height of the vertical pole from base (m)
        pole_radius: Radius of the vertical pole (m)
        ring_radius: Inner radius of the ring hole at the top (m)
        ring_thickness: Thickness of the ring material (m)
        density: Material density (kg/m³)
        color: RGBA color array, defaults to beige/tan
        initial_pose: Initial pose of the actor
        
    Returns:
        The built SAPIEN actor entity
    """
    if color is None:
        color = np.array([0.87, 0.72, 0.53, 1.0])  # Beige/tan color from image
    
    builder = scene.create_actor_builder()
    
    # Material for rendering
    material = sapien.render.RenderMaterial(
        base_color=color,
        metallic=0.1,
        roughness=0.8,
    )
    
    # Physical material
    physical_material = sapien.physx.PhysxMaterial(
        static_friction=0.6,
        dynamic_friction=0.5,
        restitution=0.1,
    )
    
    # Build the square base
    base_pose = sapien.Pose(p=[0, 0, base_thickness / 2])
    
    builder.add_box_visual(
        pose=base_pose,
        half_size=[base_size / 2, base_size / 2, base_thickness / 2],
        material=material,
    )
    
    builder.add_box_collision(
        pose=base_pose,
        half_size=[base_size / 2, base_size / 2, base_thickness / 2],
        material=physical_material,
        density=density,
    )
    
    # Build the legs (extending from corners/edges of the base)
    # 3 legs positioned at 120 degrees apart
    leg_length = base_size * 0.6  # Legs extend outward from base
    leg_radius = 0.005
    # Build the three tripod legs
    leg_angles = [0, 2*np.pi/3, 4*np.pi/3]  # 120 degrees apart
    
    for angle in leg_angles:
        # Position at base_radius from center
        x = base_size/2 * np.cos(angle)
        y = base_size/2 * np.sin(angle)
        z = base_thickness-leg_length / 2  # Center of leg
        
        # Calculate the tilt angle (legs point inward and downward)
        tilt = np.arctan2(base_size/2, leg_length)
        
        # Rotation: tilt toward center, then rotate around z
        quat = _euler_to_quat(0, tilt, angle)
        
        pose = sapien.Pose(p=[x, y, z], q=quat)
        
        builder.add_cylinder_visual(
            pose=pose,
            radius=leg_radius,
            half_length=leg_length / 2,
            material=material,
        )
        
        builder.add_cylinder_collision(
            pose=pose,
            radius=leg_radius,
            half_length=leg_length / 2,
            material=physical_material,
            density=density,
        )
        
        # Add small spheres at leg tips for stability
        tip_z = - leg_length * np.cos(tilt) + base_thickness/2
        tip_pose = sapien.Pose(p=[x, y, tip_z])
        
        builder.add_sphere_visual(
            pose=tip_pose,
            radius=leg_radius * 1.5,
            material=material,
        )
        
        builder.add_sphere_collision(
            pose=tip_pose,
            radius=leg_radius * 1.5,
            material=physical_material,
            density=density,
        )
                

    
    # Build the vertical pole
    pole_center_z = base_thickness + pole_height / 2
    pole_pose = sapien.Pose(p=[0, 0, pole_center_z],q=[0.707, 0, -0.707, 0])
    
    builder.add_cylinder_visual(
        pose=pole_pose,
        radius=pole_radius,
        half_length=pole_height / 2,
        material=material,
    )
    
    builder.add_cylinder_collision(
        pose=pole_pose,
        radius=pole_radius,
        half_length=pole_height / 2,
        material=physical_material,
        density=density,
    )
    
    # Build the ring at the top (torus approximation using cylinders arranged in a circle)
    ring_center_z = base_thickness + pole_height + ring_thickness/2 + ring_radius
    num_segments = 16
    segment_angle = 2 * np.pi / num_segments
    segment_length = 2 * np.pi * ring_radius / num_segments
    
    for i in range(num_segments):
        angle = i * segment_angle
        x = ring_radius * np.cos(angle)
        y = ring_radius * np.sin(angle)
        
        # Rotation to align cylinder segment tangent to circle
        quat = _euler_to_quat(0, 0, 0)
        
        pose = sapien.Pose(
            p=[0, x, ring_center_z+y],
            q=quat
        )
        
        builder.add_cylinder_visual(
            pose=pose,
            radius=ring_thickness / 2,
            half_length=segment_length / 2,
            material=material,
        )
        
        builder.add_cylinder_collision(
            pose=pose,
            radius=ring_thickness / 2,
            half_length=segment_length / 2,
            material=physical_material,
            density=density,
        )
    
    # Set initial pose
    if initial_pose is None:
        initial_pose = sapien.Pose(p=[0, 0, 0])
    builder.initial_pose = initial_pose
    
    # Build and return the actor
    return builder.build(name=name)


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (in radians) to quaternion [w, x, y, z].
    
    Args:
        roll: Rotation around x-axis
        pitch: Rotation around y-axis  
        yaw: Rotation around z-axis
        
    Returns:
        Quaternion as [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])
