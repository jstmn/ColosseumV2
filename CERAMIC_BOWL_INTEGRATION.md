# PlaceDishInRack Plate Geometry Notes

## Overview
`PlaceDishInRack-v1` now ships with a procedurally generated dinner plate that has a raised rim designed for reliable robot grasps. The previous experiment that loaded the ceramic bowl OBJ has been rolled back because the concave collision mesh became difficult to scale and align with the rack slots.

## Key Geometry
- **Outer diameter**: 0.17 m (`_plate_outer_radius = 0.085`)
- **Inner opening**: 0.12 m (`_plate_inner_radius = 0.06`)
- **Base thickness**: 8 mm (`_plate_base_thickness = 0.008`)
- **Rim height**: 18 mm (`_plate_rim_height = 0.018`)
- **Total height**: 26 mm (`_plate_total_height`)
- Collision model: flat cylinder for the base + 96 thin boxes that approximate a circular rim so grippers can catch the edge without snagging.
- Visual model: smooth STL asset (`mani_skill/assets/dish_into_rack/procedural_plate.stl`) generated from the same dimensions.

## Implementation Highlights
- See `_build_plate()` inside `mani_skill/envs/tasks/tabletop/place_dish_in_rack.py`.
- `_plate_spawn_buffer` now keeps the plate flush with the tabletop so it starts already resting on the surface.
- Initial spawn has been shifted to `(-0.28, -0.26)` in table coordinates so the enlarged plate clears the Panda hand at reset.
- The analytic collision approximation avoids the need for convex decomposition and keeps the object fully dynamic.
- Update `_plate_outer_radius`, `_plate_inner_radius`, `_plate_base_thickness`, and `_plate_rim_height` if you need a different plate size.

## Testing Tips
Activate your ManiSkill environment and run either manual control or the planner:

```bash
python -m mani_skill.examples.motionplanning.panda.run -e PlaceDishInRack-v1
```

For a quick load smoke test without rendering:

```bash
python3 test_ceramic_bowl.py
```

The script prints the spawned actor pose; the plate should appear roughly centered at `z ≈ 0.013` after settling.

## Assets
- Procedural plate STL: `mani_skill/assets/dish_into_rack/procedural_plate.stl`
- The legacy OBJ (`mani_skill/assets/dish_into_rack/ceramic_bowl.obj`) is still preserved if you want to experiment with the high-poly bowl, but it is no longer referenced by the environment by default.
