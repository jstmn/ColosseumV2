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
from transforms3d.quaternions import quat2mat
from transforms3d.euler import mat2euler
from sapien.render import RenderBodyComponent
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose


class TextureManager:
    """Handles texture-related operations for distractor objects."""
    
    @staticmethod
    def get_random_texture(texture_dir):
        """Get a random texture from the specified directory."""
        texture_files = [f for f in os.listdir(texture_dir) if f.endswith('.png')]
        texture_file = np.random.choice(texture_files)
        return sapien.render.RenderTexture2D(filename=os.path.join(texture_dir, texture_file))

    @staticmethod
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
            texture = TextureManager.get_random_texture(textures_directory)
            
            for render_shape in render_body_component.render_shapes:
                for part in render_shape.parts:
                    part.material.set_base_color_texture(texture)
                    
            print(f"Applied texture to object {obj.name}")


class DistractorRandomization:
    """Handles randomization of object properties for distractors."""
    
    @staticmethod
    def randomize_cylinder_properties(config):
        """Generate random properties for a cylinder based on config."""
        cylinder_config = config.get("cylinder")
        
        # Get random size within range
        radius = random.uniform(*cylinder_config.get("radius_range"))
        height = random.uniform(*cylinder_config.get("height_range"))
        color = np.random.uniform(*cylinder_config.get("color_range")).tolist() + [1.0]
        
        # Generate random rotation around Y-axis
        rotation_range = cylinder_config.get("rotation_range")
        y_rotation = random.uniform(*rotation_range)
        rotation_quat = euler2quat(0, y_rotation, 0)  # Rotate around y-axis
        
        # Size for overlap checking
        obj_size = [radius*2, radius*2, height]
        
        return {
            "radius": radius,
            "height": height,
            "color": color,
            "y_rotation": y_rotation,
            "rotation_quat": rotation_quat,
            "obj_size": obj_size
        }
    
    @staticmethod
    def randomize_sphere_properties(config):
        """Generate random properties for a sphere based on config."""
        sphere_config = config.get("sphere")
        
        # Get random size within range
        radius = random.uniform(*sphere_config.get("radius_range"))
        color = np.random.uniform(*sphere_config.get("color_range")).tolist() + [1.0]
        
        obj_size = [radius*2, radius*2, radius*2]
        
        return {
            "radius": radius,
            "color": color,
            "obj_size": obj_size
        }


class CollisionDetection:
    """Handles collision detection between objects."""
    
    @staticmethod
    def calculate_bounding_box(position, size, rotation=None):
        """
        Calculate the axis-aligned bounding box for an object.
        
        Args:
            position: [x, y, z] position of the object center
            size: [width, length, height] dimensions of the object
            rotation: Optional quaternion rotation to apply
            
        Returns:
            Dictionary with min_x, max_x, min_y, max_y bounds
        """
        # Default half-extents
        half_width = size[0] / 2
        half_length = size[1] / 2
        
        # If rotation is provided for non-cylindrical objects, adjust the bounding box
        if rotation is not None:
            if isinstance(rotation, (list, tuple, np.ndarray)) and len(rotation) == 4:
                rot_matrix = quat2mat(rotation)
                _, _, yaw = mat2euler(rot_matrix)
                
                # For simplicity, we'll just consider yaw (rotation around z-axis)
                # which affects the x-y footprint
                cos_yaw = abs(np.cos(yaw))
                sin_yaw = abs(np.sin(yaw))
                
                # Rotated rectangle dimensions
                rotated_half_width = half_width * cos_yaw + half_length * sin_yaw
                rotated_half_length = half_width * sin_yaw + half_length * cos_yaw
                
                half_width, half_length = rotated_half_width, rotated_half_length
        
        # Calculate bounds
        return {
            "min_x": position[0] - half_width,
            "max_x": position[0] + half_width,
            "min_y": position[1] - half_length,
            "max_y": position[1] + half_length
        }
    
    @staticmethod
    def check_xy_overlap(pos1, size1, pos2, size2, obj_id="", rotation1=None, rotation2=None):
        """
        Check if two objects overlap in the x-y plane using bounding boxes.
        
        Args:
            pos1, pos2: Positions of the objects
            size1, size2: Sizes of the objects [width, length, height]
            obj_id: ID of the second object (for safety margin adjustment)
            rotation1, rotation2: Quaternion rotations for the objects (optional)
        """
        # Calculate safety margin - larger for manipulation object
        safety_margin = 0.02 if obj_id == "manipulation_object" else 0.015
        
        # Calculate bounding boxes for both objects
        bb1 = CollisionDetection.calculate_bounding_box(pos1, size1, rotation1)
        bb2 = CollisionDetection.calculate_bounding_box(pos2, size2, rotation2)
        
        # Add safety margin to the bounding box
        bb1["min_x"] -= safety_margin
        bb1["max_x"] += safety_margin
        bb1["min_y"] -= safety_margin
        bb1["max_y"] += safety_margin
        
        bb2["min_x"] -= safety_margin
        bb2["max_x"] += safety_margin
        bb2["min_y"] -= safety_margin
        bb2["max_y"] += safety_margin
        
        # Check for overlap - boxes overlap if they overlap on both axes
        x_overlap = (bb1["min_x"] <= bb2["max_x"] and bb1["max_x"] >= bb2["min_x"])
        y_overlap = (bb1["min_y"] <= bb2["max_y"] and bb1["max_y"] >= bb2["min_y"])
        
        return x_overlap and y_overlap

    @staticmethod
    def is_position_valid(position, size, existing_objects, rotation=None):
        """Check if a position is valid (no overlaps with existing objects)."""
        # Check against each existing object
        for obj in existing_objects:
            # Get rotation of existing object if available
            obj_rotation = obj.get('rotation')
            
            # If overlap is detected with any object, position is invalid
            if CollisionDetection.check_xy_overlap(
                position, size, 
                obj["position"], obj["size"],
                obj_id=obj["id"],
                rotation1=rotation,
                rotation2=obj_rotation
            ):
                return False
        
        return True


class DistractorPlacement:
    """Handles positioning and placement of distractor objects."""
    
    @staticmethod
    def get_valid_position_on_table(obj_size, existing_objects, max_attempts, rotation=None, obj_type=None, table_bounds=None):
        """
        Find a valid position on the table with no physical overlaps. Simple random sampling approach.
        
        Args:
            obj_size: Size of the object to place [width, length, height]
            existing_objects: List of already placed objects
            max_attempts: Maximum number of attempts to find a valid position
            rotation: Optional quaternion rotation to apply to the object
            obj_type: Type of object ("cylinder" or "sphere") for specialized positioning
            table_bounds: Optional dictionary with x_min, x_max, y_min, y_max defining table boundaries
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
        
        # Set search boundaries for the table
        x_min = table_bounds.get("x_min")
        x_max = table_bounds.get("x_max")
        y_min = table_bounds.get("y_min")
        y_max = table_bounds.get("y_max")

        # Simple random sampling with overlap checking
        for attempt in range(max_attempts):
            # Generate random position within bounds
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            position = [x, y, obj_z]
            
            # Check validity against all existing objects
            if CollisionDetection.is_position_valid(position, obj_size, existing_objects, rotation=rotation):
                return position
            
        return None

    @staticmethod
    def create_cylinder(scene, properties, position, textures_directory=None, cylinder_index=0):
        """Create a cylinder with the given properties and position it."""
        # Create cylinder
        # Note: Cylinders are initially built horizontally (z-axis aligned)
        # and half_length parameter represents half the length along z-axis
        cylinder = actors.build_cylinder(
            scene,
            initial_pose=sapien.Pose(),
            name=f"distractor_cylinder_{cylinder_index}",
            radius=properties["radius"],
            half_length=properties["height"]/2,  # half the full height
            color=properties["color"]
        )
        
        # Stand cylinder upright, with rotation around Y-axis
        # First, stand cylinder upright (rotate 90 degrees around X-axis)
        # This changes cylinder orientation from z-axis to y-axis alignment
        upright_quat = euler2quat(np.pi/2, 0, 0)  
        
        # Apply upright rotation first
        pose = sapien.Pose(p=position, q=upright_quat)
        
        # Then apply Y-rotation around the now-vertical axis
        # For cylinders, we need to apply rotation in the local coordinate frame
        final_pose = pose * sapien.Pose(p=[0, 0, 0], q=properties["rotation_quat"])
        cylinder.set_pose(final_pose)
        
        # Apply either color or texture
        # If texture directory is available, we might apply texture instead of color
        if textures_directory:
            # Randomly decide whether to use texture or color (keep the color set above)
            use_texture = random.random() < 0.5
            if use_texture:
                # Apply texture if configured
                TextureManager.apply_texture_to_objects([cylinder._objs[0]], textures_directory)
        
        # Create object data for tracking
        cylinder_data = {
            "object": cylinder,
            "position": position,
            "upright_rotation": upright_quat,
            "y_rotation": properties["rotation_quat"],
            "size": properties["obj_size"]
        }
        
        return cylinder_data
    
    @staticmethod
    def create_sphere(scene, properties, position, textures_directory=None, sphere_index=0):
        """Create a sphere with the given properties and position it."""
        sphere = actors.build_sphere(
            scene,
            initial_pose=sapien.Pose(),
            name=f"distractor_sphere_{sphere_index}",
            radius=properties["radius"],
            color=properties["color"]
        )
        
        # Set pose immediately
        pose = sapien.Pose(p=position)
        sphere.set_pose(pose)
        
        # Apply either color or texture
        # If texture directory is available, we might apply texture instead of color
        if textures_directory:
            # Randomly decide whether to use texture or color (keep the color set above)
            use_texture = random.random() < 0.5
            if use_texture:
                # Apply texture if configured
                TextureManager.apply_texture_to_objects([sphere._objs[0]], textures_directory)
        
        # Create object data for tracking
        sphere_data = {
            "object": sphere,
            "position": position,
            "size": properties["obj_size"]
        }
        
        return sphere_data


class EnhancedDistractorManager:
    """Main class for managing enhanced distractors in the scene."""
    
    @staticmethod
    def create_enhanced_distractors(scene, manipulation_obj_pos, cfg, manipulation_object_size=None):
        """
        Create enhanced distractor objects (cylinders and spheres) and place them on the table.
        
        Args:
            scene: The SAPIEN scene
            manipulation_obj_pos: Position of the manipulation object
            cfg: Configuration for enhanced distractors
            manipulation_object_size: The size of the manipulation object for overlap checking
        
        Returns:
            List of internal objects data
        """
        
        full_size = manipulation_object_size * 2
        manipulation_obj_size = [full_size, full_size, full_size]

        # Create list to track objects we've placed
        existing_objects = [
            {"position": manipulation_obj_pos, "size": manipulation_obj_size, "id": "manipulation_object"}
        ]
        
        # Create storage for objects
        internal_objects = []
        
        # Place distractors with an alternating approach to ensure variety
        max_objects_to_place = cfg.get("max_objects")  # Configurable number of objects
        cylinder_count = cfg.get("cylinder").get("count")
        sphere_count = cfg.get("sphere"
        ).get("count")
        
        max_attempts_per_object = cfg.get("max_attempts")
        textures_directory = cfg.get("textures_directory")
        
        # Get table bounds from config
        table_bounds = cfg.get("table_bounds")
        
        # First place cylinders
        for i in range(cylinder_count):
            # Get random cylinder properties
            cylinder_props = DistractorRandomization.randomize_cylinder_properties(cfg)
            
            # Try to find a valid position
            position = DistractorPlacement.get_valid_position_on_table(
                cylinder_props["obj_size"], existing_objects, 
                max_attempts=max_attempts_per_object,
                rotation=cylinder_props["rotation_quat"],
                obj_type="cylinder",  # Pass object type for proper height adjustment
                table_bounds=table_bounds
            )
            
            if position is not None:
                # Create and position cylinder with index i
                cylinder_data = DistractorPlacement.create_cylinder(
                    scene, cylinder_props, position, 
                    textures_directory, cylinder_index=i
                )
                
                # Add to tracking list
                existing_objects.append({
                    "position": position,
                    "size": cylinder_props["obj_size"],
                    "id": f"cylinder_{i}",
                    "upright_rotation": cylinder_data["upright_rotation"],
                    "y_rotation": cylinder_data["y_rotation"]
                })
                
                # Store for later
                internal_objects.append(cylinder_data)
            
        # Then place spheres
        for i in range(sphere_count):
            # Get random sphere properties
            sphere_props = DistractorRandomization.randomize_sphere_properties(cfg)
            
            # No rotation needed for spheres, they're symmetrical
            position = DistractorPlacement.get_valid_position_on_table(
                sphere_props["obj_size"], existing_objects, 
                max_attempts=max_attempts_per_object,
                obj_type="sphere",  # Pass object type for proper height adjustment
                table_bounds=table_bounds
            )
            
            if position is not None:
                # Create and position sphere with index i
                sphere_data = DistractorPlacement.create_sphere(
                    scene, sphere_props, position,
                    textures_directory, sphere_index=i
                )
                
                # Add to tracking list
                existing_objects.append({
                    "position": position,
                    "size": sphere_props["obj_size"],
                    "id": f"sphere_{i}"
                })
                
                # Store for later
                internal_objects.append(sphere_data)
        
        return internal_objects

    @staticmethod
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
