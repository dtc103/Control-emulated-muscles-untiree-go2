from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.envs.mdp.actions import JointAction
from muscle_model import MuscleModel

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
    from . import actions_cfg

class MuscleJointAction(JointAction):
    def __init__(self, cfg, env, muscle_params, options):
        super().__init__(cfg, env)
        self.muscles = MuscleModel(muscle_params=muscle_params,
                                   action_tensor=self._processed_actions, 
                                   nenvironments=self.num_envs, 
                                   options=options)

    def apply_actions(self):
        ### TODO APPLY MUSCLE MODEL HERE

        self.muscles.compute_torques(self._asset._joint_pos_target_sim, self._asset._joint_vel_target_sim)
        ##########
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)

    @property
    def action_dim(self):
        return 2 * self._num_joints # 2 times joint, so we can apply the 
    
    @property
    def raw_actions(self):
        return self._raw_actions
    


