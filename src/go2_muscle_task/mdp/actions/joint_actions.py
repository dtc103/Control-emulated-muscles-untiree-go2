from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.envs.mdp.actions import JointAction
from go2_muscle_task.actuators.actuator_muscle import MuscleModel

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
    from . import actions_cfg

class MuscleJointAction(JointAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        muscle_params = {
            "lmin":0.24,
            "lmax":1.53,
            "fvmax": 1.38,
            "fpmax": 1.76,
            "lce_min": 0.74,
            "lce_max": 0.94,
            "phi_min": -3.14,
            "phi_max": 3.14,
            "vmax": 30.0, # TODO pierre fragen, ob das das ricthige ist (taken from unitre.py UNITREE_GO2_CFG)
            "peak_force": 45.0,
            "eps": 10e-5 # eps is just a smal number for numerical purpouses
        }

        self.muscles = MuscleModel(muscle_params=muscle_params,
                                   action_tensor=self._processed_actions, 
                                   nenvironments=self.num_envs, 
                                   options=None)

    def apply_actions(self):
        torques = self.muscles.compute_torques(self._asset.data.joint_pos, self._asset.data.joint_vel, self._processed_actions)
        print(torques)
        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)

    @property
    def action_dim(self):
        return 2 * self._num_joints # 2 times joint, so we can apply the 
    
    @property
    def raw_actions(self):
        return self._raw_actions
    


