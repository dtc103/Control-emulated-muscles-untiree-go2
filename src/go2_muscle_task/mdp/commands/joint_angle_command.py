# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import JointAngleCommandCfg


class JointAngleCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    """

    cfg: JointAngleCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: JointAngleCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: [joint_pos_1, ..., joint_pos_n]
        print(self.cfg.ranges)
        self.position_command_b = torch.zeros(self.num_envs, len(self.cfg.ranges.joint_angles), device=self.device)

        # -- metrics
        self.metrics["error_joint_angles"] = torch.zeros(self.num_envs, len(self.cfg.ranges.joint_angles), device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.position_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_joint_angles"] += (
            torch.norm(self.position_command_b - self.robot.data.joint_pos) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample position commands
        #vector of length environments
        random_vec = torch.empty(len(env_ids), device=self.device)
        for i in range(0, len(self.cfg.ranges.joint_angles)):
            self.position_command_b[env_ids, i] = random_vec.uniform_(*self.cfg.ranges.joint_angles[i])


    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_position_visualizer"):
                # -- goal
                self.goal_position_visualizer = VisualizationMarkers(self.cfg.goal_position_visualizer_cfg)
                # -- current
                self.current_position_visualizer = VisualizationMarkers(self.cfg.current_position_visualizer_cfg)
            # set their visibility to true
            self.goal_position_visualizer.set_visibility(True)
            self.current_position_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_position_visualizer"):
                self.goal_position_visualizer.set_visibility(False)
                self.current_position_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # # check if robot is initialized
        # # note: this is needed in-case the robot is de-initialized. we can't access the data
        # if not self.robot.is_initialized:
        #     return
        # # get marker location
        # # -- base state
        # #self.robot.data.joint_pos
        # base_pos_w = self.robot.data.root_pos_w.clone()
        # base_pos_w[:, 2] += 0.5
        # # -- resolve the scales and quaternions
        # vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        # vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # # display markers
        # self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        # self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        pass


