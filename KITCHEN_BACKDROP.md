# Kitchen Backdrop for Dish Rack Tasks

## Overview

A realistic kitchen environment backdrop has been created for the dish rack manipulation tasks, inspired by AI2THOR's kitchen scenes. This provides a more realistic context for household robotics tasks compared to the simple table scene.

## Features

### Kitchen Scene Components

The `KitchenSceneBuilder` creates a basic kitchen counter area with:

1. **Kitchen Counter** (1.2m x 0.6m x 0.9m)
   - Granite/marble-look countertop surface (light gray)
   - Dark wood cabinet base
   - Realistic materials with proper metallic/roughness properties

2. **Backsplash** (wall behind counter)
   - Off-white tile appearance
   - Prevents objects from falling off the back
   - 0.6m height above counter

3. **Upper Cabinet**
   - Dark wood material matching base cabinets
   - Positioned 60cm above counter surface
   - Adds visual realism to kitchen scene

### New Environment

**PlaceDishInRackKitchen-v1**
- Same task as PlaceDishInRack-v1 but with kitchen backdrop
- Objects positioned on kitchen counter instead of table
- Robot positioned at appropriate height (0.7m base) to reach counter
- Counter surface at 0.9m height (standard kitchen counter height)

## File Structure

```
mani_skill/
├── utils/scene_builder/
│   └── kitchen.py                    # Kitchen scene builder
└── envs/tasks/tabletop/
    └── place_dish_in_rack_kitchen.py # Kitchen version of task
```

## Usage

### Basic Usage

```python
import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "PlaceDishInRackKitchen-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
    num_envs=1,
)

obs, info = env.reset()
# ... your code here
```

### Using the Kitchen Scene Builder

You can use the `KitchenSceneBuilder` in your own environments:

```python
from mani_skill.utils.scene_builder.kitchen import KitchenSceneBuilder

class MyKitchenEnv(BaseEnv):
    def _load_scene(self, options: Dict):
        self.kitchen_scene = KitchenSceneBuilder(
            env=self,
            robot_init_qpos_noise=0.02,
            counter_length=1.2,     # Customize counter size
            counter_width=0.6,
            counter_height=0.9,
            add_backsplash=True,    # Optional backsplash
            add_upper_cabinet=True, # Optional upper cabinet
        )
        self.kitchen_scene.build()

        # Access counter height
        counter_top_z = self.kitchen_scene.counter_top_height
```

## Key Differences from Table Scene

| Aspect | Table Scene | Kitchen Scene |
|--------|-------------|---------------|
| Work Surface Height | ~0.0m | 0.9m (standard counter) |
| Robot Base Height | 0.0m | 0.7m |
| Visual Complexity | Simple table | Counter + backsplash + cabinet |
| Materials | Basic colors | Realistic granite, wood, tile |
| Backdrop | None | Kitchen wall and cabinet |

## AI2THOR Inspiration

While AI2THOR uses Unity for rendering and physics, we've created a SAPIEN-compatible kitchen scene that captures the essential elements:

- Realistic kitchen counter dimensions
- Proper material appearances (granite countertop, wood cabinets, tile backsplash)
- Functional layout similar to AI2THOR's kitchen scenes
- Appropriate positioning for dish-related tasks

## Testing

Run these scripts to test the kitchen environment:

```bash
# View the kitchen backdrop
python test_kitchen_backdrop.py

# Visualize kitchen layout
python visualize_kitchen.py

# Test with motion planning solution
python test_kitchen_with_solution.py
```

## Customization

### Adjust Counter Dimensions

```python
kitchen_scene = KitchenSceneBuilder(
    env=self,
    counter_length=1.5,  # Wider counter
    counter_width=0.7,   # Deeper counter
    counter_height=0.85, # Lower counter
)
```

### Disable Optional Elements

```python
kitchen_scene = KitchenSceneBuilder(
    env=self,
    add_backsplash=False,    # No wall
    add_upper_cabinet=False, # No upper cabinet
)
```

### Modify Materials

Edit `kitchen.py` to change materials:

```python
# Counter top material
material=sapien.render.RenderMaterial(
    base_color=[0.9, 0.9, 0.85, 1.0],  # Change color
    metallic=0.1,
    roughness=0.3,
    specular=0.5,
)
```

## Future Enhancements

Potential additions to make the scene more realistic:

1. Sink with faucet
2. Dish soap dispenser
3. Additional cabinets and drawers
4. Kitchen appliances
5. More varied counter materials
6. Textured backsplash (actual tiles)
7. Under-cabinet lighting

## Related Environments

- **PlaceDishInRack-v1**: Original version with simple table
- **PickDishFromRack-v1**: Reverse task (pick from rack)
- **PlaceDishInRackKitchen-v1**: Kitchen backdrop version (this)

## Credits

Kitchen scene design inspired by AI2THOR's realistic household environments.
Implementation uses SAPIEN physics engine and ManiSkill framework.
