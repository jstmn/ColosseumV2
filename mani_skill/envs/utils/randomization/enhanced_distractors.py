"""
Enhanced Distractors Module for ManiSkill

This module provides functionality for creating and manipulating enhanced distractor
objects in ManiSkill environments. It includes utilities for placing objects on tables,
checking overlaps, and setting up cylinders and spheres with customizable properties.
"""

import os
import random
import numpy as np
import torch
import sapien
from transforms3d.euler import euler2quat
from sapien.render import RenderBodyComponent
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose


def get_random_texture(texture_dir):
    """Get a random texture from the specified directory."""
    texture_files = [f for f in os.listdir(texture_dir) if f.endswith('.png')]
    texture_file = np.random.choice(texture_files)
    return sapien.render.RenderTexture2D(filename=os.path.join(texture_dir, texture_file))


def apply_texture_to_objects(objects, textures_directory):
    """Apply a random texture to a list of sapien objects."""
    if not os.path.exists(textures_directory):
        print(f"Warning: Textures directory {textures_directory} does not exist")
        return
        
    for obj in objects:
        # Find render component
        render_body_component = obj.find_component_by_type(RenderBodyComponent)
        if render_body_component is None:
            continue
            
        # Get a random texture
        texture = get_random_texture(textures_directory)
        
        for render_shape in render_body_component.render_shapes:
            for part in render_shape.parts:
                part.material.set_base_color_texture(texture)
                
        print(f"Applied texture to object {obj.name}")


def check_xy_overlap(pos1, size1, pos2, size2, obj_id="", rotation1=None, rotation2=None):
    """
    Check if two objects overlap in the x-y plane using bounding boxes.
    Accounts for orientation (especially for cylinders which are rotated to stand upright).
    
    Args:
        pos1, pos2: Positions of the objects
        size1, size2: Sizes of the objects [width, length, height]
        obj_id: ID of the second object (for debug purposes)
        rotation1, rotation2: Quaternion rotations for the objects (optional)
    """
    # Calculate distances in x and y directions
    dist_x = abs(pos1[0] - pos2[0])
    dist_y = abs(pos1[1] - pos2[1])
    
    # Use a larger safety margin for the manipulation object (cube)
    safety_margin = 0.07 if obj_id == "manipulation_object" else 0.015
    
    # Identify cylinders by ID
    is_cylinder1 = "cylinder" in str(obj_id)
    is_cylinder2 = "cylinder" in str(obj_id)
    
    # For cylinders, we simplify by using the radius for both dimensions
    # since a cylinder rotated around Y still has a circular base
    if is_cylinder1:
        # Cylinder's x and y dimensions are both the diameter (2*radius)
        width1 = size1[0]  # This is already diameter, 2*radius
        length1 = size1[0]  # Same as width for cylinder (circular base)
    else:
        width1, length1 = size1[0], size1[1]
        
    if is_cylinder2:
        # Cylinder's x and y dimensions are both the diameter (2*radius)
        width2 = size2[0]  # This is already diameter, 2*radius
        length2 = size2[0]  # Same as width for cylinder (circular base)
    else:
        width2, length2 = size2[0], size2[1]
    
    # For non-cylinders, adjust for rotation if provided
    if not is_cylinder1 and rotation1 is not None:
        # If we have rotation, convert to Euler angles
        if isinstance(rotation1, (list, tuple, np.ndarray)) and len(rotation1) == 4:
            from transforms3d.quaternions import quat2mat
            from transforms3d.euler import mat2euler
            
            rot_matrix = quat2mat(rotation1)
            roll, pitch, yaw = mat2euler(rot_matrix)
            
            # For simplicity, we'll just consider yaw (rotation around z-axis)
            # which affects the x-y footprint
            cos_yaw = abs(np.cos(yaw))
            sin_yaw = abs(np.sin(yaw))
            
            # Rotated rectangle dimensions
            rotated_width1 = width1 * cos_yaw + length1 * sin_yaw
            rotated_length1 = width1 * sin_yaw + length1 * cos_yaw
            
            width1, length1 = rotated_width1, rotated_length1
    
    if not is_cylinder2 and rotation2 is not None:
        # Same for the second object
        if isinstance(rotation2, (list, tuple, np.ndarray)) and len(rotation2) == 4:
            from transforms3d.quaternions import quat2mat
            from transforms3d.euler import mat2euler
            
            rot_matrix = quat2mat(rotation2)
            roll, pitch, yaw = mat2euler(rot_matrix)
            
            cos_yaw = abs(np.cos(yaw))
            sin_yaw = abs(np.sin(yaw))
            
            rotated_width2 = width2 * cos_yaw + length2 * sin_yaw
            rotated_length2 = width2 * sin_yaw + length2 * cos_yaw
            
            width2, length2 = rotated_width2, rotated_length2
    
    # Required separation in x direction
    half_width1 = width1 / 2
    half_width2 = width2 / 2
    min_dist_x = half_width1 + half_width2 + safety_margin
    
    # Required separation in y direction
    half_length1 = length1 / 2
    half_length2 = length2 / 2
    min_dist_y = half_length1 + half_length2 + safety_margin
    
    # Check if overlapping in both dimensions
    overlap = dist_x < min_dist_x and dist_y < min_dist_y
    
    return overlap


def is_position_valid(position, size, existing_objects, rotation=None):
    """Check if a position is valid (no overlaps with existing objects)."""
    # Check against each existing object
    for obj in existing_objects:
        # Get rotation of existing object if available
        obj_rotation = obj.get('rotation', None)
        
        # If overlap is detected with any object, position is invalid
        if check_xy_overlap(
            position, size, 
            obj["position"], obj["size"],
            obj_id=obj["id"],
            rotation1=rotation,
            rotation2=obj_rotation
        ):
            return False
    
    return True


def get_valid_position_on_table(obj_size, existing_objects, max_attempts=100, rotation=None, obj_type=None):
    """
    Find a valid position on the table with no physical overlaps. Simple random sampling approach.
    
    Args:
        table_pos: Position of the table
        obj_size: Size of the object to place [width, length, height]
        existing_objects: List of already placed objects
        max_attempts: Maximum number of attempts to find a valid position
        rotation: Optional quaternion rotation to apply to the object
        obj_type: Type of object ("cylinder" or "sphere") for specialized positioning
    """
    if obj_type == "cylinder":
        # For cylinders, we need to use the radius for the z-height
        # The cylinder is built horizontally and then rotated upright
        radius = obj_size[0]/2  # obj_size[0] is diameter, so half is radius
        obj_z = radius  # Z-coordinate should be at radius height
    elif obj_type == "sphere":
        # For spheres, we position them with their bottom at the table surface
        # Since spheres are symmetrical, radius is half the obj_size (diameter)
        obj_z = obj_size[0]/2  # Radius = diameter/2
    else:
        # Default case - just place at half height
        obj_z = obj_size[2]/2
    
    # Set search boundaries for the table (expanded slightly)
    x_min, x_max = -0.20, 0.20  # Expanded to 40cm x 40cm area
    y_min, y_max = -0.20, 0.20
    
    # Simple random sampling with overlap checking
    for attempt in range(max_attempts):
        # Generate random position within bounds
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        position = [x, y, obj_z]
        
        # Check validity against all existing objects
        if is_position_valid(position, obj_size, existing_objects, rotation=rotation):
            return position
        
    return None


def create_enhanced_distractors(scene, manipulation_obj_pos, cfg):
    """
    Create enhanced distractor objects (cylinders and spheres) and place them on the table.
    
    Args:
        scene: The SAPIEN scene
        table_pos: Position of the table
        manipulation_obj_pos: Position of the manipulation object
        cfg: Configuration for enhanced distractors
    
    Returns:
        Dictionary with internal objects data
    """

    manipulation_obj_size = [0.05, 0.05, 0.05]  # assumed size manipulation object (for overlap check only)

    # Create list to track objects we've placed
    existing_objects = [
        {"position": manipulation_obj_pos, "size": manipulation_obj_size, "id": "manipulation_object"}
    ]
    
    # Create storage for objects
    internal_objects = []
    
    # Place distractors with an alternating approach to ensure variety
    max_objects_to_place = cfg.get("max_objects", 4)  # Configurable number of objects
    cylinder_count = cfg.get("cylinder", {}).get("count", max_objects_to_place // 2)
    sphere_count = cfg.get("sphere", {}).get("count", max_objects_to_place // 2)
    
    placed_cylinders = 0
    placed_spheres = 0
    max_attempts_per_object = cfg.get("max_attempts", 100)
    textures_directory = cfg.get("textures_directory", None)
    
    # First place cylinders
    for i in range(cylinder_count):
        # Get cylinder config
        cylinder_config = cfg.get("cylinder", {})
        
        # Get random size within range
        radius = random.uniform(*cylinder_config.get("radius_range", (0.02, 0.04)))
        height = random.uniform(*cylinder_config.get("height_range", (0.04, 0.08)))
        color = np.random.uniform(*cylinder_config.get("color_range", ((0, 0, 0), (1, 1, 1)))).tolist() + [1.0]
        
        # Generate random rotation around Y-axis
        rotation_range = cylinder_config.get("rotation_range", (0, np.pi/2))
        y_rotation = random.uniform(*rotation_range)
        rotation_quat = euler2quat(0, y_rotation, 0)  # Rotate around y-axis
        
        # Size for overlap checking
        obj_size = [radius*2, radius*2, height]
        
        # Try to find a valid position
        position = get_valid_position_on_table(
            obj_size, existing_objects, 
            max_attempts=max_attempts_per_object,
            rotation=rotation_quat,
            obj_type="cylinder"  # Pass object type for proper height adjustment
        )
        
        if position is not None:
            # Create cylinder
            # Note: Cylinders are initially built horizontally (z-axis aligned)
            # and half_length parameter represents half the length along z-axis
            cylinder = actors.build_cylinder(
                scene,
                initial_pose=sapien.Pose(),
                name=f"distractor_cylinder_{i}",
                radius=radius,
                half_length=height/2,  # half the full height
                color=color
            )
            
            # Stand cylinder upright, with rotation around Y-axis
            # First, stand cylinder upright (rotate 90 degrees around X-axis)
            # This changes cylinder orientation from z-axis to y-axis alignment
            upright_quat = euler2quat(np.pi/2, 0, 0)  
            
            # Apply upright rotation first
            pose = sapien.Pose(p=position, q=upright_quat)
            
            # Then apply Y-rotation around the now-vertical axis
            # For cylinders, we need to apply rotation in the local coordinate frame
            final_pose = pose * sapien.Pose(p=[0, 0, 0], q=rotation_quat)
            cylinder.set_pose(final_pose)
            
            # Apply texture if configured
            if textures_directory and random.random() < cfg.get("texture_probability", 0.5):
                apply_texture_to_objects([cylinder._objs[0]], textures_directory)
            
            # Add to tracking list with separate rotation components
            existing_objects.append({
                "position": position,
                "size": obj_size,
                "id": f"cylinder_{i}",
                "upright_rotation": upright_quat,
                "y_rotation": rotation_quat
            })
            
            # Store for later with upright and y-rotation separated
            internal_objects.append({
                "object": cylinder,
                "position": position,
                "upright_rotation": upright_quat,
                "y_rotation": rotation_quat,
                "size": obj_size
            })
            
            placed_cylinders += 1
        
    # Then place spheres
    for i in range(sphere_count):
        # Get sphere config
        sphere_config = cfg.get("sphere", {})
        
        # Get random size within range
        radius = random.uniform(*sphere_config.get("radius_range", (0.02, 0.04)))
        color = np.random.uniform(*sphere_config.get("color_range", ((0, 0, 0), (1, 1, 1)))).tolist() + [1.0]
        
        obj_size = [radius*2, radius*2, radius*2]
        
        # No rotation needed for spheres, they're symmetrical
        position = get_valid_position_on_table(
            obj_size, existing_objects, 
            max_attempts=max_attempts_per_object,
            obj_type="sphere"  # Pass object type for proper height adjustment
        )
        
        if position is not None:
            sphere = actors.build_sphere(
                scene,
                initial_pose=sapien.Pose(),
                name=f"distractor_sphere_{i}",
                radius=radius,
                color=color
            )
            
            # Set pose immediately
            pose = sapien.Pose(p=position)
            sphere.set_pose(pose)
            
            # Apply texture if configured
            if textures_directory and random.random() < cfg.get("texture_probability", 0.5):
                apply_texture_to_objects([sphere._objs[0]], textures_directory)
            
            existing_objects.append({
                "position": position,
                "size": obj_size,
                "id": f"sphere_{i}"
            })
            
            internal_objects.append({
                "object": sphere,
                "position": position,
                "size": obj_size
            })
            
            placed_spheres += 1
    
    return internal_objects


def position_enhanced_distractors(internal_objects, n_envs):
    """
    Set the positions of enhanced distractor objects for all environments.
    
    Args:
        internal_objects: List of internal object data
        n_envs: Number of environments
    """
    for i, obj_data in enumerate(internal_objects):
        # Create tensor with position replicated for all environments
        pos = torch.tensor(
            obj_data["position"], 
            dtype=torch.float32
        ).repeat(n_envs, 1)
        
        # If this is a cylinder with separate rotation components
        if "upright_rotation" in obj_data and "y_rotation" in obj_data:
            # Get the rotation quaternions
            upright_q = obj_data["upright_rotation"]
            y_rotation_q = obj_data["y_rotation"]
            
            # Create tensor versions for all environments
            upright_qt = torch.tensor(upright_q, dtype=torch.float32).repeat(n_envs, 1)
            y_rotation_qt = torch.tensor(y_rotation_q, dtype=torch.float32).repeat(n_envs, 1)
            
            # First set the upright pose
            upright_pose = Pose.create_from_pq(p=pos, q=upright_qt)
            # Then calculate the final pose with y-rotation
            y_rotation_pose = Pose.create_from_pq(p=torch.zeros((n_envs, 3), dtype=torch.float32), q=y_rotation_qt)
            
            # Apply both rotations
            obj_data["object"].set_pose(upright_pose * y_rotation_pose)
        # Handle objects with just position (spheres)
        else:
            # Set pose without rotation
            obj_data["object"].set_pose(Pose.create_from_pq(p=pos)) 