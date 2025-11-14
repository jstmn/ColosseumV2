#!/usr/bin/env python3
"""Test motion planning for the ceramic bowl in PlaceDishInRack environment"""

import argparse
from mani_skill.examples.motionplanning.panda.run import main

if __name__ == "__main__":
    # Run motion planning for PlaceDishInRack with the ceramic bowl
    import sys
    sys.argv = [
        "test_bowl_motion_planning.py",
        "-e", "PlaceDishInRack-v1",
        "-n", "1",  # 1 episode
        "--render-mode", "human",  # Show visualization
        "--vis",  # Visualize grasp poses
        # "--save-video",  # Uncomment to save video
    ]
    main()
