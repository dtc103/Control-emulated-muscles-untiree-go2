# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import POSITION_GOAL_MARKER_CFG, CUBOID_MARKER_CFG
from omni.isaac.lab.utils import configclass

from .joint_angle_command import JointAngleCommand

@configclass
class JointAngleCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = JointAngleCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""


    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        joint_angles: list[tuple[float, float]] = MISSING

    ranges: Ranges = MISSING


    goal_position_visualizer_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(
        prim_path="/Visuals/Command/position_goal"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""
    current_position_visualizer_cfg: VisualizationMarkersCfg = POSITION_GOAL_MARKER_CFG.replace(
        prim_path="/Visuals/Command/position_current"
    )



